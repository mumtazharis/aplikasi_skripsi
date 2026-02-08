import cv2
import time
from collections import deque
import numpy as np
import mediapipe as mp
from PySide6.QtCore import QThread, Signal

class CameraWorker(QThread):
    frame_ready = Signal(np.ndarray)
    frame_ml = Signal(np.ndarray)
    camera_info = Signal(int, int, float)
    current_fps = Signal(float)

    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self.running = True
        self.flip_horizontal = False

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,           
            refine_landmarks=True,     
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

    def set_flip(self, value: bool):
        self.flip_horizontal = value

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_setting = cap.get(cv2.CAP_PROP_FPS)

        # Kirim sekali ke UI
        self.camera_info.emit(width, height, fps_setting)

        # FPS
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

            frame_base = frame.copy()
            frame_ui = frame.copy()   # untuk digambar
            frame_ml = frame.copy()   # untuk ML (BERSIH)

            h, w, _ = frame_base.shape

            # MediaPipe pakai RGB
            rgb = cv2.cvtColor(frame_base, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)

            # Gambar bounding box HANYA ke frame_ui
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:

                    # Ambil semua titik landmark (x,y)
                    xs = [int(lm.x * w) for lm in face_landmarks.landmark]
                    ys = [int(lm.y * h) for lm in face_landmarks.landmark]

                    # Hitung bounding box dari landmark
                    x_min, x_max = max(0, min(xs)), min(w, max(xs))
                    y_min, y_max = max(0, min(ys)), min(h, max(ys))

                    # Gambar bounding box ke UI SAJA
                    cv2.rectangle(
                        frame_ui,
                        (x_min, y_min),
                        (x_max, y_max),
                        (0, 255, 0),
                        2
                    )

            # FPS
            now = time.time()
            dt = now - prev_time
            prev_time = now

            if dt > 0:
                instant_fps = 1.0 / dt
                fps_buffer.append(instant_fps)

            # Emit FPS tiap 0.5 detik (BUKAN tiap frame)
            if now - last_fps_emit >= fps_update_interval and fps_buffer:
                avg_fps = sum(fps_buffer) / len(fps_buffer)
                self.current_fps.emit(avg_fps)
                last_fps_emit = now

            self.frame_ready.emit(frame_ui)
            self.frame_ml.emit(frame_ml)

            self.msleep(1)

        cap.release()

    def stop(self):
        self.running = False
        self.quit()
        self.wait()
