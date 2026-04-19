import mediapipe as mp
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
import bisect
import gc

# ==========================================
# 1. KONFIGURASI
# ==========================================
INPUT_DIR = r"..\Compressed_version2_wholeDataset\part_A"
OUTPUT_DIR_FLOW = "output_flow_enriched"       # (T, 63) enriched time-series ROI flow vectors
OUTPUT_DIR_STRAIN = "output_strain"   # (T, 9, 64, 64, 1) time-series strain patches
ANNOTATION_PATH = r"Book2.xlsx"
TARGET_ALIGN_SIZE = (320, 320)
REGION_SIZE = (64, 64)

# --- ROI INDICES (SAMA PERSIS DENGAN SPOTTING.PY) ---
# Termasuk area_pangkal_hidung untuk kompensasi gerakan kepala
ROI_INDICES = {
    "area_dahi": [109, 104, 333, 338],
    "area_alis_kanan": [104, 46, 55, 107],
    "area_alis_kiri": [333, 276, 285, 336],
    "area_antara_alis": [151, 55, 168, 285],
    "area_pipi_kanan": [114, 117, 216, 98],
    "area_pipi_kiri": [343, 346, 436, 294],
    "area_hidung": [114, 164, 343, 168],
    "area_mulut_kanan": [216, 169, 200, 0],
    "area_mulut_kiri": [436, 394, 200, 0],
    "area_pangkal_hidung": [168, 193, 195, 417],
}

# ROI yang DISIMPAN ke output (9 ROI, tanpa mata, tanpa pangkal hidung)
ROI_ORDER = [
    "area_dahi",
    "area_alis_kanan", "area_alis_kiri", "area_antara_alis",
    "area_pipi_kanan", "area_pipi_kiri",
    "area_hidung",
    "area_mulut_kanan", "area_mulut_kiri"
]

LEFT_EYE_IDX = [33, 133, 159, 145]
RIGHT_EYE_IDX = [362, 263, 386, 374]
EAR_LEFT_EYE = [33, 160, 158, 133, 153, 144]
EAR_RIGHT_EYE = [362, 385, 387, 263, 373, 380]
BLINK_THRESHOLD = 0.22

# Init RAFT
print("Loading RAFT model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Menggunakan device: {device}")
weights = Raft_Large_Weights.DEFAULT
transforms = weights.transforms()
raft_model = raft_large(weights=weights, progress=False).to(device)
raft_model.eval()
print("RAFT model loaded!")

# Init MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# ==========================================
# 2. HELPER FUNCTIONS (DARI SPOTTING.PY)
# ==========================================
def detect_face(image_rgb):
    results = face_mesh.process(image_rgb)
    if not results.multi_face_landmarks: return None
    face_landmarks = results.multi_face_landmarks[0]
    h, w, _ = image_rgb.shape
    points = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks.landmark])
    return (int(np.min(points[:, 0])), int(np.min(points[:, 1])),
            int(np.max(points[:, 0]) - np.min(points[:, 0])),
            int(np.max(points[:, 1]) - np.min(points[:, 1])))

def compute_ear(landmarks, eye_indices):
    """Eye Aspect Ratio untuk deteksi kedipan (dari spotting.py)"""
    p1 = np.array([landmarks.landmark[eye_indices[0]].x, landmarks.landmark[eye_indices[0]].y])
    p2 = np.array([landmarks.landmark[eye_indices[1]].x, landmarks.landmark[eye_indices[1]].y])
    p3 = np.array([landmarks.landmark[eye_indices[2]].x, landmarks.landmark[eye_indices[2]].y])
    p4 = np.array([landmarks.landmark[eye_indices[3]].x, landmarks.landmark[eye_indices[3]].y])
    p5 = np.array([landmarks.landmark[eye_indices[4]].x, landmarks.landmark[eye_indices[4]].y])
    p6 = np.array([landmarks.landmark[eye_indices[5]].x, landmarks.landmark[eye_indices[5]].y])
    d1 = np.linalg.norm(p2 - p6)
    d2 = np.linalg.norm(p3 - p5)
    d3 = np.linalg.norm(p1 - p4)
    return (d1 + d2) / (2.0 * d3)

def detect_eye_positions_and_blink(image_rgb, landmarks):
    """Deteksi posisi mata + blink status (dari spotting.py)"""
    h, w, _ = image_rgb.shape
    # Gunakan HANYA sudut mata (canthus) yang stabil untuk alignment
    left_eye = np.mean([[landmarks.landmark[i].x * w, landmarks.landmark[i].y * h] for i in [33, 133]], axis=0)
    right_eye = np.mean([[landmarks.landmark[i].x * w, landmarks.landmark[i].y * h] for i in [362, 263]], axis=0)
    ear_left = compute_ear(landmarks, EAR_LEFT_EYE)
    ear_right = compute_ear(landmarks, EAR_RIGHT_EYE)
    is_blinking = (ear_left < BLINK_THRESHOLD) or (ear_right < BLINK_THRESHOLD)
    return left_eye, right_eye, is_blinking

def get_square_box(box, img_shape, margin=0.15):
    h_img, w_img = img_shape[:2]
    x, y, w, h = box
    center_x, center_y = x + w // 2, y + h // 2
    max_dim = max(w, h)
    side_length = int(max_dim * (1 + margin * 2))
    new_x = max(0, center_x - side_length // 2)
    new_y = max(0, center_y - side_length // 2)
    if new_x + side_length > w_img: new_x = max(0, w_img - side_length)
    if new_y + side_length > h_img: new_y = max(0, h_img - side_length)
    final_w = min(side_length, w_img - new_x)
    final_h = min(side_length, h_img - new_y)
    return new_x, new_y, final_w, final_h

def align_and_crop_face(bgr_img, target_size=(320, 320), enable_align=True,
                        smooth_state=None, alpha=0.15, alpha_size=0.02):
    """
    Versi spotting.py — dengan blink detection dan decoupled alpha_size.
    Alignment tidak di-update saat kedipan agar tidak terdistorsi.
    """
    if bgr_img is None: return None, smooth_state

    if smooth_state is None:
        smooth_state = {'angle': 0.0, 'center': None, 'box': None, 'is_blinking': False}

    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_img)
    if not results.multi_face_landmarks: return None, smooth_state
    landmarks = results.multi_face_landmarks[0]

    if enable_align:
        left_eye, right_eye, is_blinking = detect_eye_positions_and_blink(rgb_img, landmarks)
        smooth_state['is_blinking'] = is_blinking

        dx, dy = right_eye[0] - left_eye[0], right_eye[1] - left_eye[1]
        raw_angle = np.degrees(np.arctan2(dy, dx))
        raw_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)

        if smooth_state['center'] is None:
            smooth_state['angle'], smooth_state['center'] = raw_angle, raw_center
        else:
            if not is_blinking:
                smooth_state['angle'] = (alpha * raw_angle) + ((1 - alpha) * smooth_state['angle'])
                smooth_state['center'] = (
                    (alpha * raw_center[0]) + ((1 - alpha) * smooth_state['center'][0]),
                    (alpha * raw_center[1]) + ((1 - alpha) * smooth_state['center'][1])
                )

        h, w = bgr_img.shape[:2]
        M = cv2.getRotationMatrix2D(smooth_state['center'], smooth_state['angle'], 1.0)
        processed_bgr = cv2.warpAffine(bgr_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    else:
        processed_bgr = bgr_img.copy()
        smooth_state['is_blinking'] = False

    # Ekstrak face_box dari gambar hasil rotasi
    rotated_results = face_mesh.process(cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB))
    if rotated_results.multi_face_landmarks:
        rot_lms = rotated_results.multi_face_landmarks[0]
        h_rot, w_rot = processed_bgr.shape[:2]
        rot_points = np.array([(lm.x * w_rot, lm.y * h_rot) for lm in rot_lms.landmark])
        face_box = (int(np.min(rot_points[:, 0])), int(np.min(rot_points[:, 1])),
                    int(np.max(rot_points[:, 0]) - np.min(rot_points[:, 0])),
                    int(np.max(rot_points[:, 1]) - np.min(rot_points[:, 1])))
    else:
        return None, smooth_state

    if smooth_state['box'] is None:
        smooth_state['box'] = face_box
    else:
        if not smooth_state['is_blinking']:
            x, y, w_box, h_box = face_box
            px, py, pw, ph = smooth_state['box']

            # DECOUPLE UKURAN: alpha_size sangat kecil agar skala stabil
            new_w = (alpha_size * w_box) + ((1 - alpha_size) * pw)
            new_h = (alpha_size * h_box) + ((1 - alpha_size) * ph)

            # DECOUPLE POSISI: center-based smoothing
            curr_cx = x + (w_box / 2.0)
            curr_cy = y + (h_box / 2.0)
            prev_cx = px + (pw / 2.0)
            prev_cy = py + (ph / 2.0)

            new_cx = (alpha * curr_cx) + ((1 - alpha) * prev_cx)
            new_cy = (alpha * curr_cy) + ((1 - alpha) * prev_cy)

            new_x = int(new_cx - (new_w / 2.0))
            new_y = int(new_cy - (new_h / 2.0))

            smooth_state['box'] = (new_x, new_y, int(new_w), int(new_h))

    sq_x, sq_y, sq_w, sq_h = get_square_box(smooth_state['box'], processed_bgr.shape, margin=0.15)
    cropped = processed_bgr[sq_y:sq_y + sq_h, sq_x:sq_x + sq_w]
    if cropped.size == 0: return None, smooth_state

    return cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA), smooth_state

def compute_optical_flow(prev_bgr, curr_bgr, model=raft_model, device=device, transforms=transforms):
    """Menghitung optical flow menggunakan RAFT (sama dengan spotting.py)"""
    prev_rgb = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2RGB)
    curr_rgb = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2RGB)
    h_asli, w_asli = prev_rgb.shape[:2]
    img1_tensor = torch.from_numpy(prev_rgb).permute(2, 0, 1).unsqueeze(0)
    img2_tensor = torch.from_numpy(curr_rgb).permute(2, 0, 1).unsqueeze(0)
    img1_batch, img2_batch = transforms(img1_tensor, img2_tensor)
    img1_batch = img1_batch.to(device)
    img2_batch = img2_batch.to(device)
    with torch.no_grad():
        list_of_flows = model(img1_batch, img2_batch, num_flow_updates=24)
        predicted_flow = list_of_flows[-1]
    flow_numpy = predicted_flow[0].permute(1, 2, 0).cpu().numpy()
    flow_numpy = flow_numpy[:h_asli, :w_asli]
    return flow_numpy

def get_dominant_movement(matrix):
    """Ambil pergerakan dominan (persentil terjauh dari nol)"""
    if matrix.size == 0: return 0.0
    p95 = np.percentile(matrix, 95)
    p5 = np.percentile(matrix, 5)
    if abs(p95) > abs(p5):
        return float(p95)
    else:
        return float(p5)

def get_roi_dominant_flow(flow, landmarks, img_w, img_h):
    """Mengekstrak arah gerakan (u, v) dominan beserta rentang std-nya untuk setiap ROI"""
    roi_flows = {}
    for name, indices in ROI_INDICES.items():
        pts = np.array([[int(landmarks.landmark[i].x * img_w),
                         int(landmarks.landmark[i].y * img_h)] for i in indices])
        x_min, y_min = np.min(pts, axis=0)
        x_max, y_max = np.max(pts, axis=0)
        if name == "area_dahi":
            roi_h = y_max - y_min
            y_min = max(0, int(y_min - roi_h * 0.6))
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(img_w, x_max), min(img_h, y_max)
        crop_flow = flow[y_min:y_max, x_min:x_max]
        if crop_flow.size == 0:
            roi_flows[name] = (0.0, 0.0, 0.0, 0.0)
        else:
            du = get_dominant_movement(crop_flow[..., 0])
            dv = get_dominant_movement(crop_flow[..., 1])
            std_u = np.std(crop_flow[..., 0])
            std_v = np.std(crop_flow[..., 1])
            roi_flows[name] = (du, dv, float(std_u), float(std_v))
    return roi_flows

def compute_strain_from_flow(flow, landmarks, img_w, img_h):
    """
    Hitung optical strain dari flow, dengan kompensasi pangkal hidung.
    Sama dengan compute_optical_strain() versi lama, tapi menerima flow langsung.
    """
    nose_indices = ROI_INDICES["area_pangkal_hidung"]
    pts = np.array([[int(landmarks.landmark[i].x * img_w),
                     int(landmarks.landmark[i].y * img_h)] for i in nose_indices])
    x_min, y_min = np.min(pts, axis=0)
    x_max, y_max = np.max(pts, axis=0)
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(img_w, x_max), min(img_h, y_max)

    crop_flow = flow[y_min:y_max, x_min:x_max]
    global_du = get_dominant_movement(crop_flow[..., 0])
    global_dv = get_dominant_movement(crop_flow[..., 1])

    u = flow[..., 0] - global_du
    v = flow[..., 1] - global_dv

    ux = cv2.Sobel(u, cv2.CV_32F, 1, 0, ksize=3)
    uy = cv2.Sobel(u, cv2.CV_32F, 0, 1, ksize=3)
    vx = cv2.Sobel(v, cv2.CV_32F, 1, 0, ksize=3)
    vy = cv2.Sobel(v, cv2.CV_32F, 0, 1, ksize=3)

    exx, eyy = ux, vy
    exy = 0.5 * (uy + vx)
    strain = np.sqrt(exx**2 + eyy**2 + 2 * exy**2)
    return strain.astype(np.float32)

def extract_regions_preserved(strain_img, landmarks, img_w, img_h):
    """Extract 9 region patches dari strain map (tanpa mata), dengan letterbox padding."""
    regions = {}
    target_w, target_h = REGION_SIZE

    for name in ROI_ORDER:
        indices = ROI_INDICES[name]
        pts = np.array([[int(landmarks.landmark[i].x * img_w),
                         int(landmarks.landmark[i].y * img_h)] for i in indices])
        x_min, y_min = np.min(pts, axis=0)
        x_max, y_max = np.max(pts, axis=0)

        roi_h = y_max - y_min
        if name == "area_dahi":
            extra_top = int(roi_h * 0.6)
            y_min = max(0, y_min - extra_top)

        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(img_w, x_max), min(img_h, y_max)

        crop = strain_img[y_min:y_max, x_min:x_max]

        if crop.size == 0:
            regions[name] = np.zeros((target_h, target_w), dtype=np.float32)
            continue

        h_c, w_c = crop.shape[:2]
        scale = min(target_w / w_c, target_h / h_c)
        new_w, new_h = int(w_c * scale), int(h_c * scale)
        resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_w = target_w - new_w
        pad_h = target_h - new_h
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=0.0)
        regions[name] = padded.astype(np.float32)

    return regions

# ==========================================
# 3. FOLDER-LEVEL PROCESSING (OPTIMIZED)
# ==========================================
def process_folder(folder_path, frame_files, annotation_ranges):
    """
    Align SELURUH folder (murah, MediaPipe) untuk temporal smoothing yang benar,
    tapi hitung RAFT flow HANYA untuk frame pairs dalam annotation ranges (mahal).

    Args:
        folder_path: path ke folder frame
        frame_files: sorted list of filenames
        annotation_ranges: list of (onset_num, offset_num) tuples

    Returns:
        valid_frame_numbers: list of int (nomor frame asli yang berhasil di-align)
        cached_flows: dict {idx: flow_array} — sparse, hanya indices yang dibutuhkan
        cached_landmarks: dict {idx: landmarks} — sparse, hanya indices yang dibutuhkan
    """
    # Step 1: Align SELURUH folder (murah — hanya MediaPipe, menjaga temporal smoothing)
    frames_aligned = []
    valid_frame_numbers = []
    current_smooth_state = None

    print("  [1/3] Aligning ALL frames (MediaPipe)...")
    for f in tqdm(frame_files, desc="    Align", leave=False):
        img = cv2.imread(os.path.join(folder_path, f))
        if img is None:
            continue
        aligned, current_smooth_state = align_and_crop_face(
            img, TARGET_ALIGN_SIZE, enable_align=True, smooth_state=current_smooth_state
        )
        if aligned is not None:
            frames_aligned.append(aligned)
            file_num = int(''.join(filter(str.isdigit, f)) or 0)
            valid_frame_numbers.append(file_num)

    if len(frames_aligned) < 2:
        return None, None, None

    # Step 2: Tentukan indices mana yang butuh flow computation
    # flow[i] = flow dari frames_aligned[i] ke frames_aligned[i+1]
    # Untuk annotation (onset, offset), butuh flow indices dari onset_idx sampai offset_idx-1
    flow_needed = set()
    for onset_num, offset_num in annotation_ranges:
        onset_idx = bisect.bisect_left(valid_frame_numbers, onset_num)
        offset_idx = bisect.bisect_right(valid_frame_numbers, offset_num) - 1
        for i in range(onset_idx, min(offset_idx, len(frames_aligned) - 1)):
            flow_needed.add(i)

    print(f"  [2/3] Computing RAFT flow for {len(flow_needed)}/{len(frames_aligned)-1} pairs...")

    # Step 3: Compute flow + landmarks HANYA untuk indices yang dibutuhkan
    cached_flows = {}   # sparse dict
    cached_landmarks = {}  # sparse dict

    for i in tqdm(sorted(flow_needed), desc="    Flow", leave=False):
        if i >= len(frames_aligned) - 1:
            continue

        # Compute flow (MAHAL - RAFT)
        flow = compute_optical_flow(frames_aligned[i], frames_aligned[i + 1])
        cached_flows[i] = flow

        # Detect landmarks pada frame "from" (untuk ROI extraction)
        if i not in cached_landmarks:
            frame_rgb = cv2.cvtColor(frames_aligned[i], cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)
            if results.multi_face_landmarks:
                cached_landmarks[i] = results.multi_face_landmarks[0]
            else:
                cached_landmarks[i] = None

    print(f"  [3/3] Done! Flows: {len(cached_flows)}, Landmarks: {len(cached_landmarks)}")

    # Free aligned frames dari memory
    del frames_aligned
    gc.collect()
    torch.cuda.empty_cache()

    return valid_frame_numbers, cached_flows, cached_landmarks


def extract_sample(cached_flows, cached_landmarks, onset_idx, offset_idx):
    """
    Extract satu sample time-series dari onset_idx sampai offset_idx.
    cached_flows dan cached_landmarks sekarang berupa dict (sparse).

    Returns:
        flow_series:   np.array shape (T, 18) — 9 ROI × 2 (du, dv), kompensasi pangkal hidung
        strain_series: np.array shape (T, 9, 64, 64, 1) — 9 region strain patches
    """
    h, w = TARGET_ALIGN_SIZE[1], TARGET_ALIGN_SIZE[0]  # 320, 320
    flow_series = []
    strain_series = []

    for i in range(onset_idx, offset_idx):
        flow = cached_flows.get(i)
        landmarks = cached_landmarks.get(i)

        if flow is None or landmarks is None:
            # Flow/landmark tidak tersedia, isi zeros (9 ROI x 7 = 63 feature)
            flow_series.append(np.zeros(63, dtype=np.float32))
            strain_series.append(np.zeros((9, 64, 64, 1), dtype=np.float32))
            continue

        # === OUTPUT 2: Strain regions (9, 64, 64, 1) per frame ===
        # Kita esktrak strain dulu karena akan kita pakai sebagian valuenya untuk flow_series
        strain_img = compute_strain_from_flow(flow, landmarks, w, h)
        roi_crops = extract_regions_preserved(strain_img, landmarks, w, h)

        stacked_regions = []
        for key in ROI_ORDER:
            img = roi_crops[key]
            img = np.expand_dims(img, axis=-1)  # (64, 64) -> (64, 64, 1)
            stacked_regions.append(img)
        strain_series.append(np.array(stacked_regions, dtype=np.float32))

        # === OUTPUT 1: Enriched Flow vectors (63,) per frame ===
        roi_flows = get_roi_dominant_flow(flow, landmarks, w, h)
        global_data = roi_flows.get("area_pangkal_hidung", (0.0, 0.0, 0.0, 0.0))
        global_du, global_dv = global_data[0], global_data[1]

        frame_features = []
        for roi_name in ROI_ORDER:
            du, dv, std_u, std_v = roi_flows.get(roi_name, (0.0, 0.0, 0.0, 0.0))
            du_clean = du - global_du
            dv_clean = dv - global_dv
            
            # 1. Magnitude & Angle
            magnitude = np.sqrt(du_clean**2 + dv_clean**2)
            angle = np.arctan2(dv_clean, du_clean)

            # 2. Max Strain value for this ROI
            img_strain = roi_crops[roi_name]
            max_strain = float(np.max(img_strain)) if img_strain.size > 0 else 0.0

            # 7 Fitur per ROI
            frame_features.extend([du_clean, dv_clean, magnitude, angle, std_u, std_v, max_strain])
            
        flow_series.append(np.array(frame_features, dtype=np.float32))

    if len(flow_series) == 0:
        return None, None

    # flow_series: (T, 18)
    # strain_series: (T, 9, 64, 64, 1)
    return np.array(flow_series, dtype=np.float32), np.array(strain_series, dtype=np.float32)


# ==========================================
# 4. MAIN LOOP
# ==========================================
def main():
    os.makedirs(OUTPUT_DIR_FLOW, exist_ok=True)
    os.makedirs(OUTPUT_DIR_STRAIN, exist_ok=True)

    if not os.path.exists(ANNOTATION_PATH):
        print(f"File Excel tidak ditemukan: {ANNOTATION_PATH}")
        return

    annotation = pd.read_excel(ANNOTATION_PATH)
    print(f"Total annotations: {len(annotation)}")

    # Group annotations by (Subject, Filename) → satu folder diproses sekali
    annotation['_subject'] = annotation['Subject'].astype(str).str.strip()
    annotation['_filename'] = annotation['Filename'].astype(str).str.strip()
    grouped = annotation.groupby(['_subject', '_filename'])

    print(f"Total unique folders: {len(grouped)}")

    metadata_rows = []
    skip_count = 0
    success_count = 0

    for (subject, filename), group in tqdm(grouped, desc="Processing Folders"):
        folder = os.path.join(INPUT_DIR, subject, filename, "color")
        if not os.path.exists(folder):
            print(f"\n  [SKIP] Folder not found: {folder}")
            skip_count += len(group)
            continue

        # Baca daftar frame
        frame_files = sorted(
            [f for f in os.listdir(folder) if f.endswith((".jpg", ".png"))],
            key=lambda x: int(''.join(filter(str.isdigit, x)) or 0)
        )

        if len(frame_files) < 2:
            skip_count += len(group)
            continue

        # Kumpulkan annotation ranges untuk folder ini
        annotation_ranges = []
        for _, row in group.iterrows():
            annotation_ranges.append((int(row['Onset']), int(row['Offset'])))

        print(f"\n{'='*60}")
        print(f"Folder: {subject}/{filename} ({len(frame_files)} frames, {len(group)} annotations)")
        print(f"{'='*60}")

        # Proses HANYA frame yang dibutuhkan (dengan warmup)
        valid_frame_numbers, cached_flows, cached_landmarks = process_folder(
            folder, frame_files, annotation_ranges
        )

        if valid_frame_numbers is None:
            print("  [SKIP] Tidak cukup frame yang berhasil di-align")
            skip_count += len(group)
            continue

        print(f"  Valid aligned frames: {len(valid_frame_numbers)}")
        print(f"  Flow pairs computed: {len(cached_flows)}")

        # Extract per annotation
        for _, row in group.iterrows():
            onset_num = int(row['Onset'])
            offset_num = int(row['Offset'])
            label = str(row.get('emotion', 'Unknown')).strip()
            if pd.isna(label) or label == '': label = 'Unknown'

            # Cari index terdekat dalam valid_frame_numbers
            # onset_idx: frame valid pertama >= onset_num
            onset_idx = bisect.bisect_left(valid_frame_numbers, onset_num)
            # offset_idx: frame valid terakhir <= offset_num
            offset_idx = bisect.bisect_right(valid_frame_numbers, offset_num) - 1

            # Validasi range
            if onset_idx >= len(valid_frame_numbers) or offset_idx < 0:
                print(f"    [SKIP] {subject}_{filename} on={onset_num} off={offset_num}: di luar range")
                skip_count += 1
                continue

            if onset_idx >= offset_idx:
                print(f"    [SKIP] {subject}_{filename} on={onset_num} off={offset_num}: range terlalu pendek")
                skip_count += 1
                continue

            # Extract time-series
            flow_series, strain_series = extract_sample(
                cached_flows, cached_landmarks, onset_idx, offset_idx
            )

            if flow_series is None or strain_series is None:
                skip_count += 1
                continue

            actual_onset = valid_frame_numbers[onset_idx]
            actual_offset = valid_frame_numbers[min(offset_idx, len(valid_frame_numbers) - 1)]
            T = len(flow_series)

            # Save flow (T, 18)
            flow_label_dir = os.path.join(OUTPUT_DIR_FLOW, label)
            os.makedirs(flow_label_dir, exist_ok=True)
            flow_name = f"{subject}_{filename}_on{onset_num}_off{offset_num}.npy"
            flow_path = os.path.join(flow_label_dir, flow_name)
            np.save(flow_path, flow_series)

            # Save strain (T, 9, 64, 64, 1)
            strain_label_dir = os.path.join(OUTPUT_DIR_STRAIN, label)
            os.makedirs(strain_label_dir, exist_ok=True)
            strain_name = f"{subject}_{filename}_on{onset_num}_off{offset_num}.npy"
            strain_path = os.path.join(strain_label_dir, strain_name)
            np.save(strain_path, strain_series)

            metadata_rows.append({
                'subject': subject,
                'filename': filename,
                'onset': onset_num,
                'offset': offset_num,
                'actual_onset': actual_onset,
                'actual_offset': actual_offset,
                'emotion': label,
                'seq_length': T,
                'flow_path': flow_path,
                'strain_path': strain_path,
                'flow_shape': str(flow_series.shape),
                'strain_shape': str(strain_series.shape),
            })

            print(f"    [OK] {flow_name} | T={T} | flow={flow_series.shape} strain={strain_series.shape}")
            success_count += 1

        # Free memory setelah selesai satu folder
        del cached_flows, cached_landmarks
        gc.collect()
        torch.cuda.empty_cache()

    # Save metadata CSV
    if metadata_rows:
        meta_df = pd.DataFrame(metadata_rows)

        meta_flow_path = os.path.join(OUTPUT_DIR_FLOW, 'metadata.csv')
        meta_df.to_csv(meta_flow_path, index=False)
        print(f"\nMetadata flow saved: {meta_flow_path}")

        meta_strain_path = os.path.join(OUTPUT_DIR_STRAIN, 'metadata.csv')
        meta_df.to_csv(meta_strain_path, index=False)
        print(f"Metadata strain saved: {meta_strain_path}")

    print(f"\n{'='*60}")
    print(f"SELESAI")
    print(f"{'='*60}")
    print(f"Berhasil diproses: {success_count}")
    print(f"Dilewati: {skip_count}")
    print(f"\nOutput directories:")
    print(f"  Flow (T, 18):          {os.path.abspath(OUTPUT_DIR_FLOW)}")
    print(f"  Strain (T, 9, 64, 64, 1): {os.path.abspath(OUTPUT_DIR_STRAIN)}")


if __name__ == "__main__":
    main()