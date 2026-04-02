"""
workers/prediction_worker.py
=============================
Prediction Worker — 4-Phase Offline Pipeline (sesuai notebook):
  Fase 1: Align semua frame → simpan di memory
  Fase 2: Macro prediction per-frame (MobileNetV2)
  Fase 3: Micro spotting (RAFT accumulated flow energy)
  Fase 4: Micro classification per-event (CNN+Transformer on Optical Strain)
  Merge → CSV
"""

import os
import csv
import time
import cv2
import numpy as np
import torch
from PySide6.QtCore import QThread, Signal

from ml.pipeline import (
    init_models,
    align_and_crop_face,
    calculate_accumulated_flow_energy,
    merge_roi_events,
    compute_optical_strain,
    extract_regions_preserved,
    MACRO_CLASSES, MICRO_CLASSES, ROI_ORDER, TARGET_ALIGN_SIZE,
)


class PredictionWorker(QThread):
    """
    Worker thread untuk menjalankan prediksi secara offline (batch).
    Input: path video ATAU path folder berisi frame images.
    Output: CSV file berisi hasil prediksi per frame (macro + micro).
    """
    progress = Signal(int, int)       # (current_step, total_steps)
    status = Signal(str)              # Status text
    finished = Signal(str)            # csv_path saat selesai
    error = Signal(str)               # error message

    def __init__(self, source_path, output_dir="results"):
        super().__init__()
        self.source_path = source_path
        self.output_dir = output_dir
        self.running = True
        self.is_folder = os.path.isdir(source_path)

    def stop(self):
        self.running = False

    def run(self):
        try:
            os.makedirs(self.output_dir, exist_ok=True)

            # ============================================
            # INIT MODELS
            # ============================================
            self.status.emit("Memuat semua model (RAFT, Macro, Micro)...")
            models = init_models()

            face_mesh = models['face_mesh']
            raft_model = models['raft_model']
            transforms_raft = models['transforms_raft']
            model_macro = models['model_macro']
            macro_transform = models['macro_transform']
            model_micro = models['model_micro']
            device = models['device']

            if not self.running:
                return

            # ============================================
            # READ ALL FRAMES
            # ============================================
            self.status.emit("Membaca frame...")
            raw_frames_bgr = self._read_all_frames()

            if not raw_frames_bgr:
                self.error.emit("Tidak ada frame yang bisa dibaca.")
                return

            total_raw = len(raw_frames_bgr)
            self.status.emit(f"Dibaca {total_raw} frame.")

            if not self.running:
                return

            # ============================================
            # FASE 1: ALIGN SEMUA FRAME → 320x320
            # ============================================
            self.status.emit(f"Fase 1/{4}: Alignment wajah ({total_raw} frame)...")
            frames_aligned = []
            valid_indices = []
            smooth_state = None

            for i, frame_bgr in enumerate(raw_frames_bgr):
                if not self.running:
                    return

                aligned_bgr, smooth_state = align_and_crop_face(
                    frame_bgr, face_mesh,
                    target_size=TARGET_ALIGN_SIZE,
                    enable_align=True,
                    smooth_state=smooth_state
                )

                if aligned_bgr is not None:
                    frames_aligned.append(aligned_bgr)
                    valid_indices.append(i)

                if i % 10 == 0:
                    self.progress.emit(i + 1, total_raw)

            self.progress.emit(total_raw, total_raw)
            self.status.emit(f"Fase 1 selesai: {len(frames_aligned)}/{total_raw} frame berhasil di-align.")

            if not frames_aligned:
                self.error.emit("Tidak ada wajah yang terdeteksi di seluruh frame.")
                return

            if not self.running:
                return

            # ============================================
            # FASE 2: MACRO PREDICTION PER-FRAME
            # ============================================
            total_aligned = len(frames_aligned)
            self.status.emit(f"Fase 2/{4}: Prediksi Macro ({total_aligned} frame)...")

            macro_predictions = []  # list of (aligned_idx, class_name, confidence)

            for idx, aligned_bgr in enumerate(frames_aligned):
                if not self.running:
                    return

                aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
                with torch.no_grad():
                    input_tensor = macro_transform(aligned_rgb).unsqueeze(0).to(device)
                    out_macro = model_macro(input_tensor)
                    probs = torch.softmax(out_macro, dim=1)
                    conf, preds = torch.max(probs, 1)
                    pred_class = MACRO_CLASSES[preds.item()]
                    confidence = conf.item()

                macro_predictions.append((idx, pred_class, confidence))

                if idx % 10 == 0:
                    self.progress.emit(idx + 1, total_aligned)

            self.progress.emit(total_aligned, total_aligned)
            self.status.emit(f"Fase 2 selesai: {len(macro_predictions)} prediksi macro.")

            if not self.running:
                return

            # ============================================
            # FASE 3: MICRO SPOTTING (RAFT)
            # ============================================
            self.status.emit(f"Fase 3/{4}: Micro Spotting ({total_aligned - 1} pairs)...")

            def spotter_progress(current, total):
                self.progress.emit(current, total)
                if current % 10 == 0:
                    self.status.emit(
                        f"Fase 3/{4}: Optical Flow {current}/{total} pairs..."
                    )

            _, _, history_per_roi = calculate_accumulated_flow_energy(
                frames_aligned, face_mesh, raft_model, device, transforms_raft,
                progress_callback=spotter_progress
            )

            if not self.running:
                return

            detected_events = merge_roi_events(history_per_roi, fps=30)
            self.status.emit(
                f"Fase 3 selesai: {len(detected_events)} event mikro terdeteksi."
            )

            # ============================================
            # FASE 4: MICRO PREDICTION PER-EVENT
            # ============================================
            self.status.emit(
                f"Fase 4/{4}: Micro Classification ({len(detected_events)} events)..."
            )

            micro_spots = []  # list of (onset_idx, offset_idx, class_name)

            for ev_idx, ev in enumerate(detected_events):
                if not self.running:
                    return

                on_idx = ev['onset']
                ap_idx = ev['apex']
                off_idx = ev['offset']

                onset_bgr = frames_aligned[on_idx]
                apex_bgr = frames_aligned[ap_idx]

                onset_rgb = cv2.cvtColor(onset_bgr, cv2.COLOR_BGR2RGB)
                res_lm = face_mesh.process(onset_rgb)

                if not res_lm.multi_face_landmarks:
                    print(f"[Micro] Skip event {ev_idx}: FaceMesh gagal pada frame {on_idx}")
                    continue

                landmarks = res_lm.multi_face_landmarks[0]
                h, w = onset_bgr.shape[:2]

                # 1. Optical Strain
                strain_img = compute_optical_strain(
                    onset_bgr, apex_bgr, landmarks,
                    raft_model, device, transforms_raft
                )

                # 2. Extract 9 regions
                roi_crops = extract_regions_preserved(strain_img, landmarks, w, h)

                stacked_regions = []
                for key in ROI_ORDER:
                    img = roi_crops[key]
                    img = np.expand_dims(img, axis=-1)
                    stacked_regions.append(img)

                # 3. Shape [9, 64, 64, 1]
                final_array = np.array(stacked_regions, dtype=np.float32)

                # 4. Tensor → [1, 9, 1, 64, 64]
                tensor_input = torch.tensor(final_array)
                tensor_input = tensor_input.permute(0, 3, 1, 2)  # [9, 1, 64, 64]
                tensor_input = tensor_input.unsqueeze(0).to(device)  # [1, 9, 1, 64, 64]

                # 5. Predict
                with torch.no_grad():
                    out_micro = model_micro(tensor_input)
                    probs_micro = torch.softmax(out_micro, dim=1)
                    conf_micro, pred_m = torch.max(probs_micro, 1)
                    pred_class_micro = MICRO_CLASSES[pred_m.item()]
                    micro_confidence = conf_micro.item()

                micro_spots.append((on_idx, off_idx, pred_class_micro, micro_confidence))

                self.progress.emit(ev_idx + 1, len(detected_events))

            self.status.emit(
                f"Fase 4 selesai: {len(micro_spots)} prediksi mikro."
            )

            if not self.running:
                return

            # ============================================
            # MERGE & WRITE CSV
            # ============================================
            self.status.emit("Menyimpan hasil...")

            source_name = os.path.splitext(os.path.basename(self.source_path))[0]
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            csv_filename = f"{source_name}_{timestamp}.csv"
            csv_path = os.path.join(self.output_dir, csv_filename)
            meta_path = csv_path.replace(".csv", "_meta.txt")

            # Build micro label lookup: aligned_idx → (micro_class, confidence)
            micro_labels = {}
            for on_idx, off_idx, micro_class, micro_conf_val in micro_spots:
                for fi in range(on_idx, off_idx + 1):
                    micro_labels[fi] = (micro_class, micro_conf_val)

            # Get video metadata
            fps_val = 30
            vid_width, vid_height = TARGET_ALIGN_SIZE

            if not self.is_folder:
                cap = cv2.VideoCapture(self.source_path)
                if cap.isOpened():
                    fps_val = cap.get(cv2.CAP_PROP_FPS) or 30
                    vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()

            # Write CSV
            annotations = {}
            with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    "frame_index", "timestamp_ms",
                    "macro_label", "macro_confidence",
                    "micro_label", "micro_confidence",
                    "face_x1", "face_y1", "face_x2", "face_y2",
                ])

                for aligned_idx, (_, macro_class, macro_conf) in enumerate(macro_predictions):
                    raw_idx = valid_indices[aligned_idx]
                    timestamp_ms = (raw_idx / fps_val) * 1000

                    micro_entry = micro_labels.get(aligned_idx, None)
                    if micro_entry:
                        micro_label, micro_conf = micro_entry
                    else:
                        micro_label = ""
                        micro_conf = 0

                    # Face box from raw frame landmarks (approximate from aligned frame)
                    # We use the aligned index mapped to the raw frame
                    frame_bgr = raw_frames_bgr[raw_idx]
                    fh, fw = frame_bgr.shape[:2]
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(frame_rgb)

                    face_x1, face_y1, face_x2, face_y2 = 0, 0, 0, 0
                    if results.multi_face_landmarks:
                        lm = results.multi_face_landmarks[0]
                        xs = [int(l.x * fw) for l in lm.landmark]
                        ys = [int(l.y * fh) for l in lm.landmark]
                        face_x1, face_y1 = min(xs), min(ys)
                        face_x2, face_y2 = max(xs), max(ys)

                    writer.writerow([
                        raw_idx,
                        f"{timestamp_ms:.1f}",
                        macro_class,
                        f"{macro_conf:.4f}",
                        micro_label,
                        f"{micro_conf:.4f}" if micro_conf else 0,
                        face_x1, face_y1, face_x2, face_y2,
                    ])

                    annotations[raw_idx] = {
                        "macro_label": macro_class,
                        "macro_conf": macro_conf,
                        "micro_label": micro_label,
                        "face_x1": face_x1,
                        "face_y1": face_y1,
                        "face_x2": face_x2,
                        "face_y2": face_y2,
                    }

            # ============================================
            # GENERATE ANNOTATED VIDEO
            # ============================================
            self.status.emit("Membuat video hasil (annotated)...")
            video_path = csv_path.replace(".csv", "_annotated.avi")
            h, w = raw_frames_bgr[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            writer_vid = cv2.VideoWriter(video_path, fourcc, fps_val, (w, h))

            total_raw = len(raw_frames_bgr)
            for i, frame_bgr in enumerate(raw_frames_bgr):
                if not self.running:
                    writer_vid.release()
                    return
                
                out_frame = frame_bgr.copy()
                ann = annotations.get(i)
                if ann:
                    x1, y1 = ann["face_x1"], ann["face_y1"]
                    x2, y2 = ann["face_x2"], ann["face_y2"]
                    if x1 > 0 or y1 > 0 or x2 > 0 or y2 > 0:
                        label = ann["macro_label"]
                        label_lower = label.lower()
                        if label_lower == "positive":
                            color = (80, 175, 76)  # BGR green
                        elif label_lower == "negative":
                            color = (60, 76, 231)  # BGR red
                        elif label_lower == "neutral":
                            color = (23, 168, 230) # BGR amber
                        else:
                            color = (136, 136, 136)

                        cv2.rectangle(out_frame, (x1, y1), (x2, y2), color, 2)
                        
                        display_text = f"{label} ({ann['macro_conf']:.2f})"
                        micro = ann["micro_label"]
                        if micro and micro.lower() not in ('', 'n/a'):
                            display_text += f" | Micro: {micro}"
                            
                        cv2.putText(
                            out_frame, display_text,
                            (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                        )
                
                writer_vid.write(out_frame)
                
                if i % 10 == 0:
                    self.status.emit(f"Menyimpan video... {i + 1}/{total_raw}")
            
            writer_vid.release()

            # Save metadata
            with open(meta_path, "w", encoding="utf-8") as f:
                f.write(f"source={os.path.basename(video_path)}\n")
                f.write(f"is_folder=False\n")
                f.write(f"fps={fps_val}\n")
                f.write(f"width={w}\n")
                f.write(f"height={h}\n")

            # Zip the package
            import zipfile
            
            self.status.emit("Membuat paket hasil (.result)...")
            zip_path = csv_path.replace(".csv", ".result")
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(csv_path, arcname=os.path.basename(csv_path))
                zipf.write(video_path, arcname=os.path.basename(video_path))
                zipf.write(meta_path, arcname=os.path.basename(meta_path))
            
            try:
                os.remove(csv_path)
                os.remove(video_path)
                os.remove(meta_path)
            except Exception as e:
                print(f"Cleanup error: {e}")

            self.status.emit("Prediksi selesai!")
            self.finished.emit(zip_path)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))

    # ==========================================
    # HELPER: READ ALL FRAMES
    # ==========================================
    def _read_all_frames(self):
        """Baca semua frame dari video atau folder ke memory."""
        frames = []

        if self.is_folder:
            valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
            raw_files = [
                f for f in os.listdir(self.source_path)
                if os.path.splitext(f)[1].lower() in valid_ext
            ]
            files = sorted(raw_files, key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))

            for i, fname in enumerate(files):
                if not self.running:
                    return frames
                filepath = os.path.join(self.source_path, fname)
                img = cv2.imread(filepath)
                if img is not None:
                    frames.append(img)
                if i % 50 == 0:
                    self.status.emit(f"Membaca frame {i + 1}/{len(files)}...")
        else:
            cap = cv2.VideoCapture(self.source_path)
            if not cap.isOpened():
                self.error.emit(f"Gagal membuka video: {self.source_path}")
                return frames

            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            idx = 0
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                idx += 1
                if idx % 50 == 0:
                    self.status.emit(f"Membaca frame {idx}/{total}...")

            cap.release()

        return frames

