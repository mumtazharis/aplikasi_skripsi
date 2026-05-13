"""
workers/prediction_worker.py
=============================
Prediction Worker — 4-Phase Offline Pipeline (sesuai notebook):
  Fase 1: Align semua frame → simpan di memory
  Fase 2: Macro prediction per-frame (MobileNetV2)
  Fase 3: Micro spotting (RAFT accumulated flow energy)
  Fase 4: Micro classification per-event (CNN+Transformer on Optical Strain)
  Merge → CSV

Mendukung batch processing untuk video panjang agar RAM tidak penuh.
Default: 900 frame per batch (~30 detik @30fps), overlap 90 frame.
"""

import os
import csv
import time
import gc
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
    SPOTTER_ROI_INDICES,
)


class PredictionWorker(QThread):
    """
    Worker thread untuk menjalankan prediksi secara offline (batch).
    Input: path video ATAU path folder berisi frame images.
    Output: CSV file berisi hasil prediksi per frame (macro + micro).
    Mendukung batch processing untuk video panjang.
    """
    progress = Signal(int, int)       # (current_step, total_steps)
    status = Signal(str)              # Status text
    finished = Signal(str)            # csv_path saat selesai
    error = Signal(str)               # error message

    TARGET_FPS = 30
    BATCH_SIZE = 900     # 30 detik @30fps
    OVERLAP = 90         # 3 detik overlap antar batch

    def __init__(self, source_path, output_dir="results", batch_size=None):
        super().__init__()
        self.source_path = source_path
        self.output_dir = output_dir
        self.running = True
        self.is_folder = os.path.isdir(source_path)
        if batch_size is not None:
            self.BATCH_SIZE = batch_size

    def stop(self):
        self.running = False

    # ==========================================
    # HELPER: VIDEO INFO (tanpa memuat frame)
    # ==========================================
    def _get_video_info(self):
        """Pre-scan video/folder untuk mendapatkan metadata tanpa memuat frame."""
        if self.is_folder:
            files = self._get_sorted_image_files()
            if not files:
                return 0, 30, 0, 0
            sample = cv2.imread(os.path.join(self.source_path, files[0]))
            if sample is None:
                return 0, 30, 0, 0
            h, w = sample.shape[:2]
            return len(files), 30, w, h
        else:
            cap = cv2.VideoCapture(self.source_path)
            if not cap.isOpened():
                return 0, 30, 0, 0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            return total, fps, w, h

    def _get_sorted_image_files(self):
        """Return sorted list of image filenames in source folder."""
        valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        raw_files = [
            f for f in os.listdir(self.source_path)
            if os.path.splitext(f)[1].lower() in valid_ext
        ]
        return sorted(raw_files, key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))

    # ==========================================
    # HELPER: READ FRAME BATCH
    # ==========================================
    def _read_frame_batch_from_video(self, cap, start_frame, count, source_fps):
        """Baca sejumlah frame dari video mulai posisi tertentu.
        Handle resampling jika FPS berbeda dari TARGET_FPS.
        """
        needs_resample = abs(source_fps - self.TARGET_FPS) >= 0.5

        if needs_resample:
            # Hitung range frame sumber yang dibutuhkan
            # start_frame dan count sudah dalam target fps space
            src_start = int(start_frame * source_fps / self.TARGET_FPS)
            src_end = int((start_frame + count) * source_fps / self.TARGET_FPS) + 1
            src_count = src_end - src_start

            cap.set(cv2.CAP_PROP_POS_FRAMES, src_start)
            raw_frames = []
            for _ in range(src_count):
                if not self.running:
                    return []
                ret, frame = cap.read()
                if not ret:
                    break
                raw_frames.append(frame)

            # Resample ke target fps
            frames = []
            for i in range(count):
                if not self.running:
                    return frames
                source_idx = int(i * source_fps / self.TARGET_FPS)
                source_idx = min(source_idx, len(raw_frames) - 1)
                if source_idx < 0 or source_idx >= len(raw_frames):
                    break
                frames.append(raw_frames[source_idx])
            return frames
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frames = []
            for _ in range(count):
                if not self.running:
                    return frames
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            return frames

    def _read_frame_batch_from_folder(self, files, start_idx, count):
        """Baca sejumlah frame dari folder mulai index tertentu."""
        frames = []
        end_idx = min(start_idx + count, len(files))
        for i in range(start_idx, end_idx):
            if not self.running:
                return frames
            filepath = os.path.join(self.source_path, files[i])
            img = cv2.imread(filepath)
            if img is not None:
                frames.append(img)
        return frames

    # ==========================================
    # MAIN RUN — BATCH PROCESSING
    # ==========================================
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
            # GET VIDEO INFO
            # ============================================
            self.status.emit("Membaca info video...")
            total_raw_frames, source_fps, vid_width, vid_height = self._get_video_info()

            if total_raw_frames == 0:
                self.error.emit("Tidak ada frame yang bisa dibaca.")
                return

            # Hitung total frame setelah resampling
            needs_resample = not self.is_folder and abs(source_fps - self.TARGET_FPS) >= 0.5
            if needs_resample:
                duration_sec = total_raw_frames / source_fps
                total_frames = int(duration_sec * self.TARGET_FPS)
                self.status.emit(
                    f"Video: {total_raw_frames} frame @ {source_fps:.1f}fps "
                    f"→ resample ke {total_frames} frame @ {self.TARGET_FPS}fps"
                )
            else:
                total_frames = total_raw_frames
                self.status.emit(f"Total: {total_frames} frame @ {source_fps:.1f}fps")

            fps_val = self.TARGET_FPS if needs_resample else source_fps

            # Tentukan apakah perlu batch processing
            use_batching = total_frames > self.BATCH_SIZE
            if use_batching:
                num_batches = -(-total_frames // self.BATCH_SIZE)  # ceil division
                self.status.emit(
                    f"Mode batch: {num_batches} batch × {self.BATCH_SIZE} frame "
                    f"(overlap {self.OVERLAP} frame)"
                )
            else:
                num_batches = 1

            # ============================================
            # PREPARE OUTPUT PATHS
            # ============================================
            source_name = os.path.splitext(os.path.basename(self.source_path))[0]
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            csv_filename = f"{source_name}_{timestamp}.csv"
            csv_path = os.path.join(self.output_dir, csv_filename)
            video_path = csv_path.replace(".csv", "_annotated.avi")
            meta_path = csv_path.replace(".csv", "_meta.txt")

            # Open video capture (keep open across batches)
            cap = None
            folder_files = None
            if not self.is_folder:
                cap = cv2.VideoCapture(self.source_path)
                if not cap.isOpened():
                    self.error.emit(f"Gagal membuka video: {self.source_path}")
                    return
            else:
                folder_files = self._get_sorted_image_files()

            # ============================================
            # BATCH PROCESSING LOOP
            # ============================================
            # Akumulator global
            all_csv_rows = []
            roi_names = [name for name in SPOTTER_ROI_INDICES.keys()
                         if name != "area_pangkal_hidung"]

            # State carry-over antar batch
            smooth_state = None
            spotter_state = None  # initial_state untuk calculate_accumulated_flow_energy

            # Untuk video writer — buka sekali, tulis per batch
            writer_vid = None
            first_frame_shape = None

            # Global frame counter
            global_valid_indices = []
            global_macro_predictions = []
            global_micro_spots = []
            global_history_per_roi = {name: [] for name in roi_names}
            global_face_boxes = {}  # aligned_idx → (x1, y1, x2, y2)
            global_aligned_frames_count = 0

            batch_start = 0
            batch_num = 0

            while batch_start < total_frames:
                if not self.running:
                    break

                batch_num += 1
                batch_end = min(batch_start + self.BATCH_SIZE, total_frames)

                # Tentukan overlap: batch pertama tidak perlu overlap
                overlap_count = self.OVERLAP if batch_start > 0 else 0
                read_start = max(0, batch_start - overlap_count)
                read_count = batch_end - read_start

                self.status.emit(
                    f"[Batch {batch_num}/{num_batches}] "
                    f"Membaca frame {read_start}–{batch_end-1} "
                    f"({read_count} frame, overlap={overlap_count})..."
                )

                # ---- BACA FRAME BATCH ----
                if self.is_folder:
                    raw_batch = self._read_frame_batch_from_folder(
                        folder_files, read_start, read_count
                    )
                else:
                    raw_batch = self._read_frame_batch_from_video(
                        cap, read_start, read_count, source_fps
                    )

                if not raw_batch:
                    self.status.emit(f"[Batch {batch_num}] Tidak ada frame, skip.")
                    batch_start = batch_end
                    continue

                # ---- FASE 1: ALIGNMENT ----
                self.status.emit(
                    f"[Batch {batch_num}/{num_batches}] Fase 1/4: Alignment "
                    f"({len(raw_batch)} frame)..."
                )

                frames_aligned = []
                batch_valid_indices = []  # index relatif terhadap raw_batch

                for i, frame_bgr in enumerate(raw_batch):
                    if not self.running:
                        break

                    aligned_bgr, smooth_state = align_and_crop_face(
                        frame_bgr, face_mesh,
                        target_size=TARGET_ALIGN_SIZE,
                        enable_align=True,
                        smooth_state=smooth_state
                    )

                    if aligned_bgr is not None:
                        frames_aligned.append(aligned_bgr)
                        # Index global = read_start + i
                        batch_valid_indices.append(read_start + i)

                    if i % 10 == 0:
                        self.progress.emit(i + 1, len(raw_batch))

                if not self.running:
                    break

                self.status.emit(
                    f"[Batch {batch_num}] Fase 1 selesai: "
                    f"{len(frames_aligned)}/{len(raw_batch)} frame aligned."
                )

                if not frames_aligned:
                    batch_start = batch_end
                    del raw_batch
                    gc.collect()
                    continue

                # Tentukan mana frame yang termasuk overlap vs frame baru
                # Frame baru dimulai dari index global >= batch_start
                new_frame_start_local = 0
                if overlap_count > 0:
                    for idx, global_idx in enumerate(batch_valid_indices):
                        if global_idx >= batch_start:
                            new_frame_start_local = idx
                            break

                # ---- FASE 2: MACRO PREDICTION ----
                total_aligned = len(frames_aligned)
                self.status.emit(
                    f"[Batch {batch_num}/{num_batches}] Fase 2/4: Macro prediction "
                    f"({total_aligned} frame)..."
                )

                batch_macro = []
                for idx, aligned_bgr in enumerate(frames_aligned):
                    if not self.running:
                        break

                    aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
                    with torch.no_grad():
                        input_tensor = macro_transform(aligned_rgb).unsqueeze(0).to(device)
                        out_macro = model_macro(input_tensor)
                        probs = torch.softmax(out_macro, dim=1)
                        conf, preds = torch.max(probs, 1)
                        pred_class = MACRO_CLASSES[preds.item()]
                        confidence = conf.item()

                    batch_macro.append((idx, pred_class, confidence))

                    if idx % 10 == 0:
                        self.progress.emit(idx + 1, total_aligned)

                if not self.running:
                    break

                # ---- FASE 3: MICRO SPOTTING ----
                self.status.emit(
                    f"[Batch {batch_num}/{num_batches}] Fase 3/4: Micro Spotting "
                    f"({total_aligned - 1} pairs)..."
                )

                def spotter_progress(current, total):
                    self.progress.emit(current, total)

                _, _, history_per_roi_batch, spotter_state = \
                    calculate_accumulated_flow_energy(
                        frames_aligned, face_mesh, raft_model, device, transforms_raft,
                        progress_callback=spotter_progress,
                        initial_state=spotter_state
                    )

                if not self.running:
                    break

                detected_events = merge_roi_events(history_per_roi_batch, fps=30)
                self.status.emit(
                    f"[Batch {batch_num}] Fase 3 selesai: "
                    f"{len(detected_events)} event mikro."
                )

                # ---- FASE 4: MICRO CLASSIFICATION ----
                self.status.emit(
                    f"[Batch {batch_num}/{num_batches}] Fase 4/4: Micro Classification "
                    f"({len(detected_events)} events)..."
                )

                batch_micro_spots = []
                for ev_idx, ev in enumerate(detected_events):
                    if not self.running:
                        break

                    on_idx = ev['onset']
                    ap_idx = ev['apex']
                    off_idx = ev['offset']

                    # Skip event yang sepenuhnya di zona overlap
                    # (sudah diproses di batch sebelumnya)
                    if overlap_count > 0 and ap_idx < new_frame_start_local:
                        continue

                    onset_bgr = frames_aligned[on_idx]
                    apex_bgr = frames_aligned[ap_idx]

                    onset_rgb = cv2.cvtColor(onset_bgr, cv2.COLOR_BGR2RGB)
                    res_lm = face_mesh.process(onset_rgb)

                    if not res_lm.multi_face_landmarks:
                        continue

                    landmarks = res_lm.multi_face_landmarks[0]
                    h, w = onset_bgr.shape[:2]

                    strain_img = compute_optical_strain(
                        onset_bgr, apex_bgr, landmarks,
                        raft_model, device, transforms_raft
                    )

                    roi_crops = extract_regions_preserved(strain_img, landmarks, w, h)

                    stacked_regions = []
                    for key in ROI_ORDER:
                        img = roi_crops[key]
                        img = np.expand_dims(img, axis=-1)
                        stacked_regions.append(img)

                    final_array = np.array(stacked_regions, dtype=np.float32)

                    s_min = final_array.min()
                    s_max = final_array.max()
                    if s_max - s_min > 1e-8:
                        final_array = (final_array - s_min) / (s_max - s_min)
                    else:
                        final_array = np.zeros_like(final_array)

                    tensor_input = torch.tensor(final_array)
                    tensor_input = tensor_input.permute(0, 3, 1, 2)
                    tensor_input = tensor_input.unsqueeze(0).to(device)

                    with torch.no_grad():
                        out_micro = model_micro(tensor_input)
                        probs_micro = torch.softmax(out_micro, dim=1)
                        conf_micro, pred_m = torch.max(probs_micro, 1)
                        pred_class_micro = MICRO_CLASSES[pred_m.item()]
                        micro_confidence = conf_micro.item()

                    # Konversi index lokal ke global aligned index
                    global_on = global_aligned_frames_count + on_idx - new_frame_start_local
                    global_ap = global_aligned_frames_count + ap_idx - new_frame_start_local
                    global_off = global_aligned_frames_count + off_idx - new_frame_start_local

                    batch_micro_spots.append(
                        (global_on, global_ap, global_off,
                         pred_class_micro, micro_confidence)
                    )

                    self.progress.emit(ev_idx + 1, len(detected_events))

                if not self.running:
                    break

                # ---- KUMPULKAN HASIL (hanya frame baru, skip overlap) ----
                for idx in range(new_frame_start_local, len(batch_valid_indices)):
                    local_aligned_idx = idx
                    raw_global_idx = batch_valid_indices[idx]

                    _, macro_class, macro_conf = batch_macro[local_aligned_idx]

                    global_valid_indices.append(raw_global_idx)
                    global_macro_predictions.append(
                        (global_aligned_frames_count + idx - new_frame_start_local,
                         macro_class, macro_conf)
                    )

                # Kumpulkan history_per_roi (skip overlap frames)
                for name in roi_names:
                    hist = history_per_roi_batch[name]
                    # hist[0] = initial 0.0, hist[1..] = per frame pair
                    # Ambil hanya frame baru (setelah overlap)
                    start_hist = new_frame_start_local
                    if start_hist == 0:
                        global_history_per_roi[name].extend(hist)
                    else:
                        global_history_per_roi[name].extend(hist[start_hist:])

                global_micro_spots.extend(batch_micro_spots)

                # ---- TULIS VIDEO CHUNK (dengan anotasi) ----
                # Build micro label lookup sementara untuk batch ini
                batch_micro_labels = {}
                for on_i, ap_i, off_i, mc, mcf in batch_micro_spots:
                    for fi in range(on_i, off_i + 1):
                        batch_micro_labels[fi] = (mc, mcf)

                for idx in range(new_frame_start_local, len(batch_valid_indices)):
                    raw_global_idx = batch_valid_indices[idx]
                    local_raw_idx = raw_global_idx - read_start
                    if local_raw_idx < 0 or local_raw_idx >= len(raw_batch):
                        continue

                    frame_bgr = raw_batch[local_raw_idx]

                    if writer_vid is None:
                        fh, fw = frame_bgr.shape[:2]
                        first_frame_shape = (fw, fh)
                        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                        writer_vid = cv2.VideoWriter(
                            video_path, fourcc, fps_val, first_frame_shape
                        )

                    out_frame = frame_bgr.copy()

                    # Deteksi wajah + anotasi
                    local_aligned_idx = idx
                    _, macro_class, macro_conf = batch_macro[local_aligned_idx]
                    global_ai = global_aligned_frames_count + idx - new_frame_start_local
                    micro_info = batch_micro_labels.get(global_ai, None)

                    fh_det, fw_det = frame_bgr.shape[:2]
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(frame_rgb)

                    if results.multi_face_landmarks:
                        lm = results.multi_face_landmarks[0]
                        xs = [int(l.x * fw_det) for l in lm.landmark]
                        ys = [int(l.y * fh_det) for l in lm.landmark]
                        x1, y1 = min(xs), min(ys)
                        x2, y2 = max(xs), max(ys)

                        # Simpan face box untuk CSV
                        global_face_boxes[global_ai] = (x1, y1, x2, y2)

                        label_lower = macro_class.lower()
                        if label_lower == "positive":
                            color = (80, 175, 76)
                        elif label_lower == "negative":
                            color = (60, 76, 231)
                        elif label_lower == "neutral":
                            color = (23, 168, 230)
                        else:
                            color = (136, 136, 136)

                        cv2.rectangle(out_frame, (x1, y1), (x2, y2), color, 1)

                        macro_text = f"{macro_class} ({macro_conf:.2f})"
                        cv2.putText(
                            out_frame, macro_text,
                            (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                        )

                        if micro_info:
                            micro_label, micro_conf = micro_info
                            if micro_label.lower() == "positive":
                                micro_color = (80, 175, 76)
                            elif micro_label.lower() == "negative":
                                micro_color = (60, 76, 231)
                            else:
                                micro_color = (200, 200, 200)

                            micro_text = f"Micro: {micro_label} ({micro_conf:.2f})"
                            cv2.putText(
                                out_frame, micro_text,
                                (x1, y1 - 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, micro_color, 2
                            )

                    writer_vid.write(out_frame)

                # Update global counter
                new_frames_count = len(batch_valid_indices) - new_frame_start_local
                global_aligned_frames_count += new_frames_count

                self.status.emit(
                    f"[Batch {batch_num}] Selesai. "
                    f"Total frame terproses: {global_aligned_frames_count}"
                )

                # ---- FREE MEMORY ----
                del raw_batch, frames_aligned, batch_macro
                del history_per_roi_batch, detected_events, batch_micro_spots
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                batch_start = batch_end

            # ============================================
            # SELESAI SEMUA BATCH — RELEASE VIDEO
            # ============================================
            if cap is not None:
                cap.release()
            if writer_vid is not None:
                writer_vid.release()

            if not self.running:
                return

            # ============================================
            # WRITE FINAL CSV
            # ============================================
            self.status.emit("Menyimpan CSV hasil...")

            # Build micro label lookup
            micro_labels = {}
            for on_idx, ap_idx, off_idx, micro_class, micro_conf_val in global_micro_spots:
                for fi in range(on_idx, off_idx + 1):
                    micro_labels[fi] = (micro_class, micro_conf_val, on_idx, ap_idx, off_idx)

            with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                header = [
                    "frame_index", "timestamp_ms",
                    "macro_label", "macro_confidence",
                    "micro_label", "micro_confidence",
                    "event_onset_frame", "event_apex_frame", "event_offset_frame",
                    "face_x1", "face_y1", "face_x2", "face_y2",
                ] + roi_names
                writer.writerow(header)

                for aligned_idx, (_, macro_class, macro_conf) in enumerate(
                        global_macro_predictions):
                    raw_idx = global_valid_indices[aligned_idx]
                    timestamp_ms = (raw_idx / fps_val) * 1000

                    micro_entry = micro_labels.get(aligned_idx, None)
                    if micro_entry:
                        micro_label, micro_conf, ev_on, ev_ap, ev_off = micro_entry
                        raw_ev_on = global_valid_indices[ev_on] if ev_on < len(global_valid_indices) else ""
                        raw_ev_ap = global_valid_indices[ev_ap] if ev_ap < len(global_valid_indices) else ""
                        raw_ev_off = global_valid_indices[ev_off] if ev_off < len(global_valid_indices) else ""
                    else:
                        micro_label = ""
                        micro_conf = 0
                        raw_ev_on, raw_ev_ap, raw_ev_off = "", "", ""

                    # Face box dari hasil deteksi saat annotasi video
                    face_x1, face_y1, face_x2, face_y2 = global_face_boxes.get(
                        aligned_idx, (0, 0, 0, 0)
                    )

                    row = [
                        raw_idx,
                        f"{timestamp_ms:.1f}",
                        macro_class,
                        f"{macro_conf:.4f}",
                        micro_label,
                        f"{micro_conf:.4f}" if micro_conf else 0,
                        raw_ev_on, raw_ev_ap, raw_ev_off,
                        face_x1, face_y1, face_x2, face_y2,
                    ]

                    for name in roi_names:
                        hist = global_history_per_roi[name]
                        val = hist[aligned_idx] if aligned_idx < len(hist) else 0.0
                        row.append(f"{val:.4f}")

                    writer.writerow(row)

            # ============================================
            # SAVE METADATA & PACKAGE
            # ============================================
            fw, fh = first_frame_shape if first_frame_shape else (vid_width, vid_height)

            with open(meta_path, "w", encoding="utf-8") as f:
                f.write(f"source={os.path.basename(video_path)}\n")
                f.write(f"is_folder=False\n")
                f.write(f"fps={fps_val}\n")
                f.write(f"width={fw}\n")
                f.write(f"height={fh}\n")

            import zipfile

            self.status.emit("Membuat paket hasil (.result)...")
            zip_path = csv_path.replace(".csv", ".result")

            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(csv_path, arcname=os.path.basename(csv_path))
                if os.path.exists(video_path):
                    zipf.write(video_path, arcname=os.path.basename(video_path))
                zipf.write(meta_path, arcname=os.path.basename(meta_path))

            try:
                os.remove(csv_path)
                if os.path.exists(video_path):
                    os.remove(video_path)
                os.remove(meta_path)
            except Exception as e:
                print(f"Cleanup error: {e}")

            self.status.emit("Prediksi selesai!")
            self.finished.emit(zip_path)

        except Exception as e:
            import traceback
            trace_str = traceback.format_exc()
            try:
                with open("DEBUG_ERROR_LOG.txt", "w") as f:
                    f.write(trace_str)
            except:
                pass
            print(trace_str)
            self.error.emit(str(e))
