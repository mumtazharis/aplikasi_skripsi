import time
import cv2
import mediapipe as mp
from ml.first_preprocessing import FirstPreprocessing
from ml.macro_predictor import MacroExpressionPredictor
from PySide6.QtCore import QThread, Signal, QMutex, QWaitCondition
from collections import deque

class MLWorker(QThread):
    # Hasil prediksi (Label, Confidence)
    prediction_result = Signal(str, float)
    # Signal opsional: Mengirim koordinat wajah untuk digambar di UI utama
    face_detected_rect = Signal(int, int, int, int) # x1, y1, x2, y2
    prediction_speed = Signal(str)

    def __init__(self):
        super().__init__()
        self.running = True
        self.latest_frame = None
        self.new_frame_available = False
        self.mutex = QMutex()
        self.condition = QWaitCondition()

    # Slot ini akan dipanggil oleh CameraWorker signal
    def update_frame(self, frame):
        self.mutex.lock()
        # Kita hanya menyimpan frame terbaru, menimpa frame lama yang belum diproses (Drop frame logic)
        self.latest_frame = frame
        self.new_frame_available = True
        self.condition.wakeOne() # Bangunkan thread ML
        self.mutex.unlock()

    def run(self):
        # Init Model HARUS di dalam run() agar thread affinity benar
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,        
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        preprocessor = FirstPreprocessing()
        predictor = MacroExpressionPredictor("models/macro_expression.onnx")

        fps_buffer = deque(maxlen=20)
        last_emit_time = time.time()

        while self.running:
            self.mutex.lock()
            # Tunggu sampai ada frame baru
            if not self.new_frame_available:
                self.condition.wait(self.mutex)
            
            # Ambil frame dan reset flag
            if self.latest_frame is None:
                self.mutex.unlock()
                continue
                
            frame_rgb = self.latest_frame.copy() # Copy agar aman
            self.new_frame_available = False
            self.mutex.unlock()

            # --- MULAI PROSES ML ---
            start_time = time.perf_counter()
            h, w, _ = frame_rgb.shape
            
            # Perlu frame BGR untuk preprocessor (sesuai kode lama Anda yang pakai 'frame' mentah)
            # Karena input camera worker tadi sudah kita ubah ke RGB, kita balikin ke BGR untuk preprocessor
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # 1. MediaPipe
            results = face_mesh.process(frame_rgb)

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]

                # A. Kirim Koordinat Box ke UI (Opsional, agar UI bisa gambar kotak)
                xs = [int(lm.x * w) for lm in face_landmarks.landmark]
                ys = [int(lm.y * h) for lm in face_landmarks.landmark]
                self.face_detected_rect.emit(min(xs), min(ys), max(xs), max(ys))

                # B. Preprocessing
                try:
                    # Note: Pastikan preprocessor Anda menerima format yang benar (BGR/RGB)
                    face_ml_processed = preprocessor.process(frame_bgr, face_landmarks)
                    
                    if face_ml_processed is not None:
                        # Convert back to RGB for ONNX Model if needed
                        face_rgb_ml = cv2.cvtColor(face_ml_processed, cv2.COLOR_BGR2RGB)
                        
                        # C. Inference
                        # Kita tidak perlu timer interval lagi di sini karena thread ini 
                        # akan berjalan secepat mungkin secara asinkronus tanpa ganggu kamera.
                        label, conf = predictor.predict(face_rgb_ml)

                        end_time = time.perf_counter()
                        speed_ms = (end_time - start_time) * 1000

                        # Hitung FPS dari latency
                        if speed_ms > 0:
                            fps = 1000.0 / speed_ms
                            fps_buffer.append(fps)

                        now = time.time()

                        # Emit hanya setiap 0.5 detik
                        if now - last_emit_time >= 0.5 and fps_buffer:
                            avg_fps = sum(fps_buffer) / len(fps_buffer)
                            avg_ms = 1000.0 / avg_fps if avg_fps > 0 else 0

                            speed_text = f"{avg_ms:.0f} ms ({avg_fps:.0f} FPS)"
                            self.prediction_speed.emit(speed_text)

                            last_emit_time = now

                        self.prediction_result.emit(label, conf)


                except Exception as e:
                    print(f"Error ML: {e}")
            else:
                # Jika tidak ada wajah, mungkin kirim signal kosong/reset
                self.prediction_result.emit("Not Detected", 0)
                speed_text = "-"
                self.prediction_speed.emit(speed_text)
                self.face_detected_rect.emit(0, 0, 0, 0)

    def stop(self):
        self.running = False
        self.condition.wakeOne() # Bangunkan thread jika sedang tidur
        self.wait()