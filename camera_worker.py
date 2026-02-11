import cv2
import time
from collections import deque
import numpy as np
import mediapipe as mp
from PySide6.QtCore import QThread, Signal

from ml.first_preprocessing import FirstPreprocessing
from ml.macro_predictor import MacroExpressionPredictor

class CameraWorker(QThread):
    frame_ready = Signal(np.ndarray) # Frame UI
    frame_ml = Signal(np.ndarray)    # Frame ML (224x224)
    prediction_result = Signal(str, float) # (Label, Confidence)

    camera_info = Signal(int, int, float)
    current_fps = Signal(float)

    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self.running = True
        self.flip_horizontal = False

        # Init MediaPipe untuk UI (Deteksi Cepat Awal)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,       
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Init Preprocessor
        self.preprocessor = FirstPreprocessing()
        # self.predictor = MacroExpressionPredictor("models/macro_expression.onnx")
        self.predictor = MacroExpressionPredictor("models/macro_expression.onnx")
        self.inference_interval = 0.2  # 0.2 detik = 5 FPS inference
        self.last_inference_time = 0

    
    def set_flip(self, value: bool):
        self.flip_horizontal = value

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Low latency

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_setting = cap.get(cv2.CAP_PROP_FPS)

        self.camera_info.emit(width, height, fps_setting)

        # FPS Variables
        prev_time = time.time()
        fps_buffer = deque(maxlen=20)
        fps_update_interval = 0.5
        last_fps_emit = time.time()

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            if self.flip_horizontal:
                frame = cv2.flip(frame, 1)

            h, w, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 1. Deteksi Awal (Untuk UI & Referensi Sudut)
            results_original = self.face_mesh.process(frame_rgb)
            
            frame_ui = frame_rgb.copy()
            face_ml_processed = None

            if results_original.multi_face_landmarks:
                # Ambil landmark pertama
                face_landmarks = results_original.multi_face_landmarks[0]

                # A. Gambar Box UI (Visualisasi User)
                xs = [int(lm.x * w) for lm in face_landmarks.landmark]
                ys = [int(lm.y * h) for lm in face_landmarks.landmark]
                cv2.rectangle(frame_ui, (min(xs), min(ys)), (max(xs), max(ys)), (0, 255, 0), 2)

                # B. Panggil Class Preprocessing
                # Kita kirim frame asli BGR dan landmark hasil deteksi awal
                try:
                    face_ml_processed = self.preprocessor.process(frame, face_landmarks)
                except Exception as e:
                    print(f"Error Preprocessing: {e}")

                if face_ml_processed is not None:
                    # face_ml_processed adalah BGR, kita ubah ke RGB untuk Model
                    face_rgb_ml = cv2.cvtColor(face_ml_processed, cv2.COLOR_BGR2RGB)
                    
                    if prev_time - self.last_inference_time >= self.inference_interval:
                        label, conf = self.predictor.predict(face_rgb_ml)
                        self.prediction_result.emit(label, conf)
                        self.last_inference_time = prev_time

            # Hitung FPS
            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0: fps_buffer.append(1.0/dt)

            if now - last_fps_emit >= fps_update_interval and fps_buffer:
                self.current_fps.emit(sum(fps_buffer)/len(fps_buffer))
                last_fps_emit = now

            # Emit Signals
            self.frame_ready.emit(frame_ui)

            self.msleep(1)

        cap.release()

    def stop(self):
        self.running = False
        self.quit()
        self.wait()