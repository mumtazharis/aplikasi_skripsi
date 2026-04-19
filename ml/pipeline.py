"""
ml/pipeline.py
==============
Full End-to-End Pipeline: Macro + Micro Expression Detection.
Ported from end_to_end_pipeline_master.ipynb

Pipeline:
1. Face Alignment 320x320 (MediaPipe + smoothing)
2. Macro: MobileNetV2 per-frame
3. Micro Spotting: RAFT Optical Flow + Accumulated Energy
4. Micro Classification: BiLSTM+Attention on Enriched Flow Time-Series
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tv_transforms
from torchvision import models
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
import mediapipe as mp
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from scipy.signal import find_peaks
import warnings

warnings.filterwarnings('ignore')

from utils.resource_path import resource_path

# ==========================================
# 1. KONFIGURASI & KONSTANTA
# ==========================================
MICRO_WEIGHTS_PATH = resource_path("models/final_lstm_flow_model.pth")
MACRO_WEIGHTS_PATH = resource_path("models/macro_mobilenet_3class.pth")

MICRO_CLASSES = ['Negative', 'Positive']
MACRO_CLASSES = ['negative', 'neutral', 'positive']

TARGET_ALIGN_SIZE = (320, 320)
REGION_SIZE = (64, 64)

LEFT_EYE_IDX = [33, 133, 159, 145]
RIGHT_EYE_IDX = [362, 263, 386, 374]
EAR_LEFT_EYE = [33, 160, 158, 133, 153, 144]
EAR_RIGHT_EYE = [362, 385, 387, 263, 373, 380]
BLINK_THRESHOLD = 0.22

# 12 Regions untuk SPOTTER (termasuk pangkal hidung)
SPOTTER_ROI_INDICES = {
    "area_dahi": [109, 104, 333, 338],
    "area_alis_kanan": [104, 46, 55, 107],
    "area_alis_kiri": [333, 276, 285, 336],
    "area_mata_kanan": [46, 111, 114, 55],
    "area_mata_kiri": [276, 340, 343, 285],
    "area_antara_alis": [151, 55, 168, 285],
    "area_pipi_kanan": [114, 117, 216, 98],
    "area_pipi_kiri": [343, 346, 436, 294],
    "area_hidung": [114, 164, 343, 168],
    "area_mulut_kanan": [216, 169, 200, 0],
    "area_mulut_kiri": [436, 394, 200, 0],
    "area_pangkal_hidung": [168, 193, 195, 417],
}

# 9 Regions untuk STRAIN EXTRACTION (Micro Model Features)
STRAIN_ROI_INDICES = {
    "area_dahi": [109, 104, 333, 338],
    "area_alis_kanan": [104, 46, 55, 107],
    "area_alis_kiri": [333, 276, 285, 336],
    "area_antara_alis": [151, 55, 168, 285],
    "area_pipi_kanan": [114, 117, 216, 98],
    "area_pipi_kiri": [343, 346, 436, 294],
    "area_hidung": [114, 164, 343, 168],
    "area_mulut_kanan": [216, 169, 200, 0],
    "area_mulut_kiri": [436, 394, 200, 0],
}

ROI_ORDER = [
    "area_dahi",
    "area_alis_kanan", "area_alis_kiri", "area_antara_alis",
    "area_pipi_kanan", "area_pipi_kiri",
    "area_hidung",
    "area_mulut_kanan", "area_mulut_kiri"
]


# ==========================================
# 2. MICRO MODEL ARCHITECTURE (BiLSTM + Attention)
# ==========================================
# Default config — akan di-override dari checkpoint saat load
LSTM_INPUT_DIM = 63       # 9 ROI × 7 features
LSTM_HIDDEN_DIM = 128
LSTM_NUM_LAYERS = 4
LSTM_BIDIRECTIONAL = True
LSTM_DROPOUT_LSTM = 0.3
LSTM_DROPOUT_FC = 0.5
MAX_SEQ_LEN = 60


class LSTMClassifier(nn.Module):
    """
    Bidirectional LSTM untuk klasifikasi time-series flow.
    Architecture:
    - LayerNorm pada input
    - Multi-layer Bidirectional LSTM
    - Attention pooling
    - Fully connected classifier dengan dropout
    """

    def __init__(self, input_dim=LSTM_INPUT_DIM, hidden_dim=LSTM_HIDDEN_DIM,
                 num_layers=LSTM_NUM_LAYERS, num_classes=2,
                 bidirectional=LSTM_BIDIRECTIONAL,
                 dropout_lstm=LSTM_DROPOUT_LSTM, dropout_fc=LSTM_DROPOUT_FC):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Input normalization
        self.input_norm = nn.LayerNorm(input_dim)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_lstm if num_layers > 1 else 0.0
        )

        lstm_output_dim = hidden_dim * self.num_directions

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.Tanh(),
            nn.Linear(lstm_output_dim // 2, 1, bias=False)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_output_dim),
            nn.Dropout(dropout_fc),
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_fc * 0.5),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, lengths):
        """
        Args:
            x: (batch, max_seq_len, input_dim) — padded sequences
            lengths: (batch,) — actual lengths
        Returns:
            logits: (batch, num_classes)
        """
        # Input normalization
        x = self.input_norm(x)

        # Pack padded sequence
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)

        # LSTM forward
        packed_output, (h_n, c_n) = self.lstm(packed)

        # Unpack
        lstm_output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Attention pooling (mask padded positions)
        attn_weights = self.attention(lstm_output).squeeze(-1)

        # Create mask for padded positions
        max_len = lstm_output.size(1)
        mask = torch.arange(max_len, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        attn_weights = attn_weights.masked_fill(~mask, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=1)

        # Weighted sum
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_output).squeeze(1)

        # Classify
        logits = self.classifier(context)
        return logits


# ==========================================
# 3. HELPER FUNCTIONS: ALIGNMENT & BLINK
# ==========================================
def detect_face(image_rgb, face_mesh):
    results = face_mesh.process(image_rgb)
    if not results.multi_face_landmarks:
        return None
    face_landmarks = results.multi_face_landmarks[0]
    h, w, _ = image_rgb.shape
    points = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks.landmark])
    return (int(np.min(points[:, 0])), int(np.min(points[:, 1])),
            int(np.max(points[:, 0]) - np.min(points[:, 0])),
            int(np.max(points[:, 1]) - np.min(points[:, 1])))


def detect_eye_positions_and_blink(image_rgb, landmarks):
    """Detect eye positions using stable canthus points and compute blink status."""
    h, w, _ = image_rgb.shape

    # Gunakan HANYA sudut mata (canthus) yang stabil untuk alignment
    # Kiri: 33 (outer), 133 (inner) | Kanan: 362 (inner), 263 (outer)
    left_eye = np.mean([[landmarks.landmark[i].x * w, landmarks.landmark[i].y * h] for i in [33, 133]], axis=0)
    right_eye = np.mean([[landmarks.landmark[i].x * w, landmarks.landmark[i].y * h] for i in [362, 263]], axis=0)

    # Hitung EAR untuk deteksi kedipan langsung
    ear_left = compute_ear(landmarks, EAR_LEFT_EYE)
    ear_right = compute_ear(landmarks, EAR_RIGHT_EYE)
    is_blinking = (ear_left < BLINK_THRESHOLD) or (ear_right < BLINK_THRESHOLD)

    return left_eye, right_eye, is_blinking


def compute_ear(landmarks, eye_indices):
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


def get_square_box(box, img_shape, margin=0.15):
    h_img, w_img = img_shape[:2]
    x, y, w, h = box
    center_x, center_y = x + w // 2, y + h // 2
    max_dim = max(w, h)
    side_length = int(max_dim * (1 + margin * 2))
    new_x = max(0, center_x - side_length // 2)
    new_y = max(0, center_y - side_length // 2)
    if new_x + side_length > w_img:
        new_x = max(0, w_img - side_length)
    if new_y + side_length > h_img:
        new_y = max(0, h_img - side_length)
    final_w = min(side_length, w_img - new_x)
    final_h = min(side_length, h_img - new_y)
    return new_x, new_y, final_w, final_h


def align_and_crop_face(bgr_img, face_mesh, target_size=(320, 320),
                         enable_align=True, smooth_state=None, alpha=0.15, alpha_size=0.02):
    """Align wajah dan crop ke target_size. Return (cropped_bgr, smooth_state)."""
    if bgr_img is None:
        return None, smooth_state
    if smooth_state is None:
        smooth_state = {'angle': 0.0, 'center': None, 'box': None, 'is_blinking': False}

    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_img)
    if not results.multi_face_landmarks:
        return None, smooth_state
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
        
        # Bounding box mentah frame ini
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
            
            # 1. DECOUPLE UKURAN (W, H): Gunakan alpha_size yang sangat kecil agar skala stabil
            new_w = (alpha_size * w_box) + ((1 - alpha_size) * pw)
            new_h = (alpha_size * h_box) + ((1 - alpha_size) * ph)
            
            # 2. DECOUPLE POSISI: Hitung titik tengah wajah saat ini dan titik tengah sebelumnya
            curr_cx = x + (w_box / 2.0)
            curr_cy = y + (h_box / 2.0)
            
            prev_cx = px + (pw / 2.0)
            prev_cy = py + (ph / 2.0)
            
            # Gunakan alpha normal untuk mengejar pergerakan kepala (translasi)
            new_cx = (alpha * curr_cx) + ((1 - alpha) * prev_cx)
            new_cy = (alpha * curr_cy) + ((1 - alpha) * prev_cy)
            
            # Konversi kembali center dan ukuran menjadi koordinat pojok kiri atas (x, y)
            new_x = int(new_cx - (new_w / 2.0))
            new_y = int(new_cy - (new_h / 2.0))
            
            smooth_state['box'] = (new_x, new_y, int(new_w), int(new_h))
    
    sq_x, sq_y, sq_w, sq_h = get_square_box(smooth_state['box'], processed_bgr.shape, margin=0.15)
    cropped = processed_bgr[sq_y:sq_y + sq_h, sq_x:sq_x + sq_w]
    if cropped.size == 0: return None, smooth_state
    
    return cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA), smooth_state

# ==========================================
# 4. HELPER FUNCTIONS: OPTICAL FLOW & STRAIN
# ==========================================
def compute_optical_flow(prev_bgr, curr_bgr, raft_model, device, transforms_raft):
    """Compute RAFT optical flow between two BGR frames."""
    prev_rgb = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2RGB)
    curr_rgb = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2RGB)

    h_asli, w_asli = prev_rgb.shape[:2]

    img1_tensor = torch.from_numpy(prev_rgb).permute(2, 0, 1).unsqueeze(0)
    img2_tensor = torch.from_numpy(curr_rgb).permute(2, 0, 1).unsqueeze(0)

    img1_batch, img2_batch = transforms_raft(img1_tensor, img2_tensor)

    img1_batch = img1_batch.to(device)
    img2_batch = img2_batch.to(device)

    with torch.no_grad():
        list_of_flows = raft_model(img1_batch, img2_batch, num_flow_updates=24)
        predicted_flow = list_of_flows[-1]

    flow_numpy = predicted_flow[0].permute(1, 2, 0).cpu().numpy()
    flow_numpy = flow_numpy[:h_asli, :w_asli]
    return flow_numpy


def get_dominant_movement(matrix):
    if matrix.size == 0:
        return 0.0
    p95 = np.percentile(matrix, 95)
    p5 = np.percentile(matrix, 5)
    if abs(p95) > abs(p5):
        return float(p95)
    else:
        return float(p5)


def get_roi_dominant_flow(flow, landmarks, img_w, img_h):
    """Digunakan oleh Spotter."""
    roi_flows = {}
    for name, indices in SPOTTER_ROI_INDICES.items():
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
            roi_flows[name] = (0.0, 0.0)
        else:
            du = get_dominant_movement(crop_flow[..., 0])
            dv = get_dominant_movement(crop_flow[..., 1])
            roi_flows[name] = (du, dv)

    return roi_flows


def compute_strain_from_flow(flow, landmarks, img_w, img_h):
    """
    Hitung optical strain dari flow array, dengan kompensasi pangkal hidung.
    Sama dengan generator.py — menerima flow langsung (tanpa compute RAFT ulang).
    """
    nose_indices = SPOTTER_ROI_INDICES["area_pangkal_hidung"]
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
    strain = np.sqrt(exx ** 2 + eyy ** 2 + 2 * exy ** 2)
    return strain.astype(np.float32)


def get_roi_enriched_flow(flow, landmarks, img_w, img_h):
    """
    Mengekstrak arah gerakan (u, v) dominan beserta std untuk setiap ROI.
    Digunakan oleh micro classification (bukan spotter).
    Menggunakan SPOTTER_ROI_INDICES (yang mencakup semua ROI termasuk pangkal hidung).
    Return dict {name: (du, dv, std_u, std_v)} untuk semua ROI.
    """
    roi_flows = {}
    for name, indices in SPOTTER_ROI_INDICES.items():
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


def extract_micro_flow_series_from_cache(cached_flows, frames_aligned, onset_idx, offset_idx,
                                         face_mesh, max_seq_len=MAX_SEQ_LEN):
    """
    Extract enriched flow time-series dari cached_flows (re-use dari spotter).
    Sama persis dengan extract_sample() di generator.py.

    Args:
        cached_flows: list of flow arrays dari calculate_accumulated_flow_energy()
                      cached_flows[i] = flow dari frames_aligned[i] ke frames_aligned[i+1]
        frames_aligned: list of aligned BGR frames
        onset_idx: index onset dalam frames_aligned
        offset_idx: index offset dalam frames_aligned
        face_mesh: MediaPipe FaceMesh instance
        max_seq_len: truncation length

    Returns:
        flow_series: np.array shape (T, 63) — 9 ROI × 7 features, atau None jika gagal
    """
    h, w = TARGET_ALIGN_SIZE[1], TARGET_ALIGN_SIZE[0]  # 320, 320
    flow_series = []

    # Iterate frame pairs dari onset sampai offset
    for i in range(onset_idx, min(offset_idx, len(cached_flows))):
        # Ambil flow dari cache (sudah dihitung di Fase 3)
        if i >= len(cached_flows):
            flow_series.append(np.zeros(63, dtype=np.float32))
            continue

        flow = cached_flows[i]

        # Detect landmarks pada frame "from" (untuk ROI extraction)
        frame_rgb = cv2.cvtColor(frames_aligned[i], cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            flow_series.append(np.zeros(63, dtype=np.float32))
            continue

        landmarks = results.multi_face_landmarks[0]

        # Compute strain dari flow (untuk max_strain feature)
        strain_img = compute_strain_from_flow(flow, landmarks, w, h)
        roi_crops = extract_regions_preserved(strain_img, landmarks, w, h)

        # Extract enriched ROI flow features
        roi_flows = get_roi_enriched_flow(flow, landmarks, w, h)
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

            # 7 Fitur per ROI: du, dv, magnitude, angle, std_u, std_v, max_strain
            frame_features.extend([du_clean, dv_clean, magnitude, angle, std_u, std_v, max_strain])

        flow_series.append(np.array(frame_features, dtype=np.float32))

        # Truncate jika sudah mencapai max_seq_len
        if len(flow_series) >= max_seq_len:
            break

    if len(flow_series) == 0:
        return None

    return np.array(flow_series, dtype=np.float32)


def extract_regions_preserved(strain_img, landmarks, img_w, img_h):
    """Extract 9 ROI regions from strain image."""
    regions = {}
    target_w, target_h = REGION_SIZE

    for name, indices in STRAIN_ROI_INDICES.items():
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
# 5. SPOTTER FUNCTIONS
# ==========================================
def calculate_accumulated_flow_energy(frames_bgr, face_mesh, raft_model, device, transforms_raft,
                                       drift_decay=0.96, motion_threshold=0.1, blink_pad=3,
                                       progress_callback=None):
    """
    Calculate accumulated flow energy for micro-expression spotting.
    progress_callback: optional callable(current, total) for progress updates.
    """
    # Pass 1: Blinks & Landmarks
    cached_landmarks = []
    raw_blinks = []

    for curr_bgr in frames_bgr:
        curr_rgb = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(curr_rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            cached_landmarks.append(landmarks)

            ear_left = compute_ear(landmarks, EAR_LEFT_EYE)
            ear_right = compute_ear(landmarks, EAR_RIGHT_EYE)
            raw_blinks.append((ear_left < BLINK_THRESHOLD) or (ear_right < BLINK_THRESHOLD))
        else:
            cached_landmarks.append(None)
            raw_blinks.append(False)

    expanded_blinks = [False] * len(raw_blinks)
    for i, is_blinking in enumerate(raw_blinks):
        if is_blinking:
            start_idx = max(0, i - blink_pad)
            end_idx = min(len(raw_blinks), i + blink_pad + 1)
            for j in range(start_idx, end_idx):
                expanded_blinks[j] = True

    # Pass 2: Flow Energy
    total_energy = [0]
    cached_flows = []
    accum_flow = {name: np.array([0.0, 0.0]) for name in SPOTTER_ROI_INDICES.keys()}
    stationary_counter = {name: 1 for name in SPOTTER_ROI_INDICES.keys()}
    history_per_roi = {name: [0.0] for name in SPOTTER_ROI_INDICES.keys()
                       if name != "area_pangkal_hidung"}

    total_pairs = len(frames_bgr) - 1
    for i in range(total_pairs):
        prev_bgr = frames_bgr[i]
        curr_bgr = frames_bgr[i + 1]

        flow = compute_optical_flow(prev_bgr, curr_bgr, raft_model, device, transforms_raft)
        cached_flows.append(flow)

        landmarks = cached_landmarks[i]
        is_blinking_expanded = expanded_blinks[i + 1]

        instant_energy = 0
        if landmarks:
            h, w = curr_bgr.shape[:2]
            roi_flows = get_roi_dominant_flow(flow, landmarks, w, h)
            global_du, global_dv = roi_flows.get("area_pangkal_hidung", (0.0, 0.0))

            roi_magnitudes = []
            for name, (du, dv) in roi_flows.items():
                if name == "area_pangkal_hidung":
                    continue

                du_bersih = du - global_du
                dv_bersih = dv - global_dv

                if name in ["area_mata_kanan", "area_mata_kiri"]:
                    du_bersih, dv_bersih = 0.0, 0.0

                current_movement = np.hypot(du_bersih, dv_bersih)

                if current_movement > motion_threshold:
                    current_decay = 1.0
                    stationary_counter[name] = 1
                else:
                    current_decay = drift_decay ** stationary_counter[name]
                    stationary_counter[name] = min(15, stationary_counter[name] + 1)

                accum_flow[name][0] = (accum_flow[name][0] * current_decay) + du_bersih
                accum_flow[name][1] = (accum_flow[name][1] * current_decay) + dv_bersih

                mag = np.linalg.norm(accum_flow[name])
                roi_magnitudes.append(mag)
                history_per_roi[name].append(mag)

            instant_energy = np.sum(roi_magnitudes)
        else:
            for name in history_per_roi:
                history_per_roi[name].append(0.0)

        total_energy.append(instant_energy)

        if progress_callback and i % 5 == 0:
            progress_callback(i + 1, total_pairs)

    return np.array(total_energy), cached_flows, history_per_roi


def detect_multiple_expressions_raw(magnitudes, fps=30, min_prominence=0.6, min_height=0.8):
    raw_mag = np.array(magnitudes)
    p95 = np.percentile(raw_mag, 95)
    median_mag = np.median(raw_mag)

    optimal_prominence = max((p95 - median_mag) * 0.5, min_prominence)
    optimal_height = max(p95 * 0.5, min_height)
    min_width = max(int(fps * 0.1), 3)

    peaks, properties = find_peaks(
        raw_mag,
        distance=max(int(fps / 5), 6),
        prominence=optimal_prominence,
        height=optimal_height,
        width=min_width
    )

    sorted_mag = np.sort(raw_mag)
    baseline_threshold = np.mean(sorted_mag[:int(max(1, len(raw_mag) * 0.3))])
    d_mag = np.gradient(raw_mag)
    grad_std = np.std(d_mag)
    grad_thresh = max(0.5 * grad_std, 0.05)

    patience_limit = max(3, int(fps * 0.1))
    results = []

    for apex in peaks:
        onset, offset = 0, len(raw_mag) - 1

        patience_count = 0
        for i in range(apex, 0, -1):
            if d_mag[i] <= grad_thresh:
                patience_count += 1
            else:
                patience_count = 0
            if patience_count >= patience_limit or raw_mag[i] <= baseline_threshold:
                onset = min(apex, i + patience_count)
                break

        patience_count = 0
        for i in range(apex, len(raw_mag) - 1):
            if d_mag[i] >= -grad_thresh:
                patience_count += 1
            else:
                patience_count = 0
            if patience_count >= patience_limit or raw_mag[i] <= baseline_threshold:
                offset = max(apex, i - patience_count)
                break

        results.append({
            "onset": onset,
            "apex": apex,
            "offset": offset,
            "prominence": properties['prominences'][list(peaks).index(apex)]
            if 'prominences' in properties else 0
        })

    return results, raw_mag


def merge_roi_events(history_per_roi, fps=30):
    all_events = []

    for roi_name, history in history_per_roi.items():
        if np.max(history) < 0.1:
            continue
        events, _ = detect_multiple_expressions_raw(magnitudes=history, fps=fps)
        for e in events:
            e['roi'] = roi_name
            all_events.append(e)

    if not all_events:
        return []

    all_events = sorted(all_events, key=lambda x: x['onset'])

    merged_events = []
    current_event = all_events[0].copy()

    for next_event in all_events[1:]:
        if next_event['onset'] <= current_event['offset'] + 1:
            current_event['onset'] = min(current_event['onset'], next_event['onset'])
            current_event['offset'] = max(current_event['offset'], next_event['offset'])

            mag_current = history_per_roi[current_event['roi']][current_event['apex']]
            mag_next = history_per_roi[next_event['roi']][next_event['apex']]

            if mag_next > mag_current:
                current_event['apex'] = next_event['apex']
                current_event['roi'] = next_event['roi']
                current_event['prominence'] = next_event['prominence']
        else:
            merged_events.append(current_event)
            current_event = next_event.copy()

    merged_events.append(current_event)
    return merged_events


# ==========================================
# 6. MODEL INITIALIZATION
# ==========================================
def init_models(device=None):
    """
    Inisialisasi semua model ML.
    Returns: dict with keys: face_mesh, raft_model, transforms_raft,
             model_macro, macro_transform, model_micro, device
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"[Pipeline] Device: {device}")

    # MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )

    # RAFT Optical Flow
    print("[Pipeline] Loading RAFT Model...")
    weights = Raft_Large_Weights.DEFAULT
    transforms_raft = weights.transforms()
    raft_model = raft_large(weights=weights, progress=False).to(device)
    raft_model.eval()
    print("[Pipeline] RAFT Model Loaded!")

    # Macro Transform (ImageNet standard)
    macro_transform = tv_transforms.Compose([
        tv_transforms.ToPILImage(),
        tv_transforms.Resize(TARGET_ALIGN_SIZE),
        tv_transforms.ToTensor(),
        tv_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Macro Model (MobileNetV2)
    model_macro = models.mobilenet_v2(weights=None)
    in_features = model_macro.classifier[1].in_features
    model_macro.classifier[1] = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, len(MACRO_CLASSES))
    )
    model_macro = model_macro.to(device)

    if os.path.exists(MACRO_WEIGHTS_PATH):
        checkpoint_macro = torch.load(MACRO_WEIGHTS_PATH, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint_macro:
            model_macro.load_state_dict(checkpoint_macro['model_state_dict'])
        else:
            model_macro.load_state_dict(checkpoint_macro)
        model_macro.eval()
        print(f"[Pipeline] Macro Model Loaded: {MACRO_WEIGHTS_PATH}")
    else:
        print(f"[Pipeline] WARNING: Macro weights not found: {MACRO_WEIGHTS_PATH}")

    # Micro Model (BiLSTM + Attention)
    # Load config dari checkpoint jika ada
    micro_config = {
        'input_dim': LSTM_INPUT_DIM,
        'hidden_dim': LSTM_HIDDEN_DIM,
        'num_layers': LSTM_NUM_LAYERS,
        'num_classes': len(MICRO_CLASSES),
        'bidirectional': LSTM_BIDIRECTIONAL,
        'dropout_lstm': LSTM_DROPOUT_LSTM,
        'dropout_fc': LSTM_DROPOUT_FC,
    }

    if os.path.exists(MICRO_WEIGHTS_PATH):
        checkpoint = torch.load(MICRO_WEIGHTS_PATH, map_location=device, weights_only=False)
        # Override config dari checkpoint jika tersedia
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            micro_config['input_dim'] = checkpoint.get('input_dim', LSTM_INPUT_DIM)
            micro_config['hidden_dim'] = checkpoint.get('hidden_dim', LSTM_HIDDEN_DIM)
            micro_config['num_layers'] = checkpoint.get('num_layers', LSTM_NUM_LAYERS)
            micro_config['num_classes'] = checkpoint.get('num_classes', len(MICRO_CLASSES))
            micro_config['bidirectional'] = checkpoint.get('bidirectional', LSTM_BIDIRECTIONAL)

        model_micro = LSTMClassifier(**micro_config).to(device)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model_micro.load_state_dict(checkpoint['model_state_dict'])
        else:
            model_micro.load_state_dict(checkpoint)
        model_micro.eval()
        print(f"[Pipeline] Micro Model (LSTM) Loaded: {MICRO_WEIGHTS_PATH}")
        print(f"[Pipeline] LSTM Config: input={micro_config['input_dim']}, "
              f"hidden={micro_config['hidden_dim']}, layers={micro_config['num_layers']}, "
              f"bidir={micro_config['bidirectional']}")
    else:
        model_micro = LSTMClassifier(**micro_config).to(device)
        print(f"[Pipeline] WARNING: Micro weights not found: {MICRO_WEIGHTS_PATH}")

    return {
        'face_mesh': face_mesh,
        'raft_model': raft_model,
        'transforms_raft': transforms_raft,
        'model_macro': model_macro,
        'macro_transform': macro_transform,
        'model_micro': model_micro,
        'device': device,
    }
