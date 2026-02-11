# import cv2
# import numpy as np
# import mediapipe as mp

# class FirstPreprocessing:
#     def __init__(self):
#         # Indeks Landmark Mata (MediaPipe)
#         self.LEFT_EYE_IDX = [33, 133, 159, 145]
#         self.RIGHT_EYE_IDX = [362, 263, 386, 374]

#         # Inisialisasi MediaPipe Face Mesh khusus untuk Preprocessing
#         # Kita menggunakan instance terpisah agar pengaturan bisa independen
#         self.mp_face_mesh = mp.solutions.face_mesh
#         self.face_mesh = self.mp_face_mesh.FaceMesh(
#             static_image_mode=True,       # True karena kita memproses per frame "potongan"
#             max_num_faces=1,
#             refine_landmarks=True,
#             min_detection_confidence=0.5
#         )

#     def get_square_box(self, box, img_shape, margin=0.15):
#         """Helper untuk membuat kotak crop menjadi persegi dengan margin."""
#         h_img, w_img = img_shape[:2]
#         x, y, w, h = box
#         center_x, center_y = x + w // 2, y + h // 2
#         max_dim = max(w, h)
#         side_length = int(max_dim * (1 + margin * 2))
        
#         new_x = max(0, center_x - side_length // 2)
#         new_y = max(0, center_y - side_length // 2)

#         if new_x + side_length > w_img: new_x = max(0, w_img - side_length)
#         if new_y + side_length > h_img: new_y = max(0, h_img - side_length)
        
#         final_w = min(side_length, w_img - new_x)
#         final_h = min(side_length, h_img - new_y)
        
#         return int(new_x), int(new_y), int(final_w), int(final_h)

#     def process(self, frame_bgr, initial_landmarks):
#         """
#         Fungsi utama:
#         1. Terima frame asli + landmark deteksi awal (dari UI thread/worker).
#         2. Hitung rotasi.
#         3. Putar gambar.
#         4. Deteksi ulang (Re-detection).
#         5. Crop & Resize.
#         """
#         if not initial_landmarks:
#             return None

#         h, w, _ = frame_bgr.shape
#         landmarks = initial_landmarks.landmark

#         # --- 1. Hitung Sudut Rotasi (dari landmark awal) ---
#         def get_avg_point(idxs):
#             points = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in idxs])
#             return np.mean(points, axis=0)

#         left_eye = get_avg_point(self.LEFT_EYE_IDX)
#         right_eye = get_avg_point(self.RIGHT_EYE_IDX)

#         dx = right_eye[0] - left_eye[0]
#         dy = right_eye[1] - left_eye[1]
#         angle = np.degrees(np.arctan2(dy, dx))
#         eyes_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)

#         # --- 2. Putar Gambar (Affine Transform) ---
#         M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
#         aligned_bgr = cv2.warpAffine(frame_bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

#         # --- 3. RE-DETECTION (Deteksi ulang pada gambar lurus) ---
#         aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
#         results_aligned = self.face_mesh.process(aligned_rgb)

#         if not results_aligned.multi_face_landmarks:
#             return None

#         # --- 4. Hitung Crop Box ---
#         face_landmarks_aligned = results_aligned.multi_face_landmarks[0]
#         x_coords = [lm.x * w for lm in face_landmarks_aligned.landmark]
#         y_coords = [lm.y * h for lm in face_landmarks_aligned.landmark]

#         x_min, x_max = int(min(x_coords)), int(max(x_coords))
#         y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
#         box_w = x_max - x_min
#         box_h = y_max - y_min
        
#         sq_x, sq_y, sq_w, sq_h = self.get_square_box((x_min, y_min, box_w, box_h), aligned_bgr.shape, margin=0.15)
        
#         # --- 5. Crop & Resize ---
#         cropped = aligned_bgr[sq_y:sq_y + sq_h, sq_x:sq_x + sq_w]
        
#         if cropped.size == 0:
#             return None
            
#         return cv2.resize(cropped, (224, 224), interpolation=cv2.INTER_AREA)

import cv2
import numpy as np
import mediapipe as mp

class FirstPreprocessing:
    def __init__(self, smooth_factor=0.1):
        # MediaPipe Init
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

        self.LEFT_EYE_IDX = [33, 133, 159, 145]
        self.RIGHT_EYE_IDX = [362, 263, 386, 374]

        # --- VARIABEL SMOOTHING ---
        # smooth_factor (alpha):
        # 0.1 = Sangat halus tapi agak delay (seperti ada efek 'berat')
        # 0.9 = Sangat responsif tapi masih agak bergetar
        # 0.3 - 0.5 = Titik tengah yang ideal
        self.alpha = smooth_factor 
        self.prev_box = None # Menyimpan koordinat [x, y, w, h] frame sebelumnya

    def get_square_box(self, box, img_shape, margin=0.15):
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
        
        return int(new_x), int(new_y), int(final_w), int(final_h)

    def process(self, frame_bgr, initial_landmarks):
        # Jika tidak ada landmark (wajah hilang), reset smoothing
        if not initial_landmarks:
            self.prev_box = None
            return None

        h, w, _ = frame_bgr.shape
        landmarks = initial_landmarks.landmark

        # --- 1. Hitung Rotasi ---
        def get_avg_point(idxs):
            points = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in idxs])
            return np.mean(points, axis=0)

        left_eye = get_avg_point(self.LEFT_EYE_IDX)
        right_eye = get_avg_point(self.RIGHT_EYE_IDX)

        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))
        eyes_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)

        M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
        aligned_bgr = cv2.warpAffine(frame_bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        # --- 2. Re-detection ---
        aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
        results_aligned = self.face_mesh.process(aligned_rgb)

        if not results_aligned.multi_face_landmarks:
            self.prev_box = None
            return None

        # --- 3. Hitung Target Box ---
        face_landmarks_aligned = results_aligned.multi_face_landmarks[0]
        x_coords = [lm.x * w for lm in face_landmarks_aligned.landmark]
        y_coords = [lm.y * h for lm in face_landmarks_aligned.landmark]

        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        # Target Box Mentah
        target_box = [x_min, y_min, x_max - x_min, y_max - y_min]
        
        # Hitung Square Box Mentah
        sq_x, sq_y, sq_w, sq_h = self.get_square_box(target_box, aligned_bgr.shape, margin=0.15)
        current_sq_box = np.array([sq_x, sq_y, sq_w, sq_h], dtype=np.float32)

        # --- 4. TERAPKAN SMOOTHING (INTI SOLUSI) ---
        if self.prev_box is None:
            # Jika ini frame pertama, langsung pakai nilai saat ini
            self.prev_box = current_sq_box
        else:
            # Rumus EMA: New = (alpha * Current) + ((1 - alpha) * Old)
            self.prev_box = (self.alpha * current_sq_box) + ((1 - self.alpha) * self.prev_box)

        # Konversi ke integer untuk cropping
        final_x, final_y, final_w, final_h = self.prev_box.astype(int)

        # --- 5. Crop & Resize ---
        cropped = aligned_bgr[final_y:final_y + final_h, final_x:final_x + final_w]
        
        if cropped.size == 0:
            return None
            
        return cv2.resize(cropped, (224, 224), interpolation=cv2.INTER_AREA) 