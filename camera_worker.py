import cv2
import time
from collections import deque
import numpy as np
from PySide6.QtCore import QThread, Signal, QMutex, QWaitCondition


class CameraWorker(QThread):
    """
    Worker thread untuk capture frame dari kamera.
    Digunakan untuk live preview saat recording di Menu 1.
    """
    # Signal untuk UI (Video feed lancar)
    frame_ready = Signal(np.ndarray)
    camera_info = Signal(int, int, float)  # width, height, fps_setting
    current_fps = Signal(float)

    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self.running = True
        self.flip_horizontal = False

    def set_flip(self, value: bool):
        self.flip_horizontal = value

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_setting = cap.get(cv2.CAP_PROP_FPS)

        self.camera_info.emit(width, height, fps_setting)

        # FPS Variables
        prev_time = time.time()
        fps_buffer = deque(maxlen=20)
        last_fps_emit = time.time()

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            if self.flip_horizontal:
                frame = cv2.flip(frame, 1)

            # Konversi ke RGB untuk UI
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Emit ke UI
            self.frame_ready.emit(frame_rgb)

            # Hitung FPS Kamera
            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                fps_buffer.append(1.0 / dt)

            if now - last_fps_emit >= 0.5 and fps_buffer:
                self.current_fps.emit(sum(fps_buffer) / len(fps_buffer))
                last_fps_emit = now

            self.msleep(10)

        cap.release()

    def stop(self):
        self.running = False
        self.wait()