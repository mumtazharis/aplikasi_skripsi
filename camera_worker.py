import cv2
import time
from collections import deque
import numpy as np
from PySide6.QtCore import QThread, Signal, QMutex, QWaitCondition

class CameraWorker(QThread):
    # Signal untuk UI (Video feed lancar)
    frame_ready = Signal(np.ndarray) 
    # Signal khusus untuk dikirim ke MLWorker
    frame_for_ml = Signal(np.ndarray) 
    
    camera_info = Signal(int, int, float)
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
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Buffer kecil untuk low latency

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

            # Konversi ke RGB untuk UI (Qt biasanya butuh RGB)
            # Jika ingin efisien, bisa kirim BGR dan konversi di UI, 
            # tapi demi konsistensi kode lama, kita ubah di sini.
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 1. Emit ke UI (Agar video tampil mulus 30/60 FPS)
            self.frame_ready.emit(frame_rgb)
            
            # 2. Emit ke ML (Worker ML akan menangkap ini jika dia 'free')
            # Kita kirim frame asli (BGR) atau RGB tergantung kebutuhan preprocessing.
            # Kode lama Anda menggunakan frame BGR untuk preprocessing dan RGB untuk MediaPipe.
            # Mari kirim RGB agar standar.
            self.frame_for_ml.emit(frame_rgb)

            # Hitung FPS Kamera
            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0: fps_buffer.append(1.0/dt)

            if now - last_fps_emit >= 0.5 and fps_buffer:
                self.current_fps.emit(sum(fps_buffer)/len(fps_buffer))
                last_fps_emit = now

            # Sleep sangat kecil untuk mencegah CPU usage 100% pada loop kosong
            self.msleep(10) # Sesuaikan, misal 10ms ~ 100FPS cap

        cap.release()

    def stop(self):
        self.running = False
        self.wait()