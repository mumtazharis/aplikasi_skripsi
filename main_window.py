import cv2
from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout
from PySide6.QtGui import QImage

from camera_worker import CameraWorker
from components.camera_view import CameraView
from components.footer import PredictionFooter
from components.sidebar import Sidebar
from ml_worker import MLWorker

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MY-APP")
        self.resize(1000, 600)

        self.camera_on = True
        self.camera_thread = None
        self.ml_thread = None
        self.flip_horizontal = False

        # Cek kamera yang tersedia
        self.available_cameras = self.list_cameras()

        # Inisialisasi Komponen UI
        self.camera_view = CameraView()
        self.sidebar = Sidebar(self.available_cameras)
        self.footer = PredictionFooter()
        # --- Susun Layout ---
        
        # 1. Container Kiri (Kamera + Footer)
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)
        left_layout.addWidget(self.camera_view, 1) # Stretch factor 1 (mengisi sisa ruang)
        left_layout.addWidget(self.footer, 0)      # Stretch factor 0 (tinggi tetap sesuai fixedHeight)

        # 2. Layout Utama (Container Kiri + Sidebar Kanan)
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Tambahkan layout kiri ke layout utama
        main_layout.addLayout(left_layout, 4) # Kiri ambil 80% lebar
        main_layout.addWidget(self.sidebar, 1) # Kanan ambil 20% lebar

        # Hubungkan Sinyal Sidebar ke Logika MainWindow
        self.sidebar.camera_changed.connect(self.change_camera)
        self.sidebar.flip_requested.connect(self.toggle_flip)
        self.sidebar.toggle_requested.connect(self.toggle_camera)

        # Mulai Kamera Default
        default_camera = self.available_cameras[0] if self.available_cameras else 0
        self.start_system(default_camera)

    def list_cameras(self, max_tested=5):
        available = []
        for i in range(max_tested):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(i)
                cap.release()
        return available

    def start_system(self, camera_index):
        """Memulai CameraWorker dan MLWorker secara bersamaan"""
        # 1. Pastikan bersih dulu (stop jika ada yang jalan)
        self.stop_system()

        # 2. Inisialisasi Workers
        self.camera_thread = CameraWorker(camera_index)
        self.ml_thread = MLWorker()

        # Set flip awal
        self.camera_thread.set_flip(self.flip_horizontal)
        
        # --- KONEKSI SINYAL (WIRING) ---

        # A. Camera -> UI (Tampilan Video Lancar)
        self.camera_thread.frame_ready.connect(self.process_frame)
        self.camera_thread.camera_info.connect(self.sidebar.update_info)
        self.camera_thread.current_fps.connect(self.sidebar.update_current_fps)

        # B. Camera -> ML (Jembatan Data Frame)
        #    Ini yang membuat ML memproses gambar dari kamera
        self.camera_thread.frame_for_ml.connect(self.ml_thread.update_frame)

        # C. ML -> UI (Hasil Prediksi)
        self.ml_thread.prediction_result.connect(self.footer.update_prediction)
        self.ml_thread.prediction_speed.connect(self.sidebar.update_prediction_speed)
        # (Opsional) Jika ingin menggambar kotak wajah dari ML
        self.ml_thread.face_detected_rect.connect(self.camera_view.update_face_box)

        # 3. Jalankan Thread
        #    Disarankan start ML dulu agar siap menerima frame
        self.ml_thread.start()
        self.camera_thread.start()
    
    def process_frame(self, frame):
        """Ubah frame OpenCV ke QImage lalu kirim ke CameraView"""
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        self.camera_view.update_frame(image)

    def toggle_flip(self):
        self.flip_horizontal = not self.flip_horizontal
        if self.camera_thread:
            self.camera_thread.set_flip(self.flip_horizontal)

    def change_camera(self, camera_index):
        # Restart sistem dengan index baru
        self.start_system(camera_index)
        self.camera_on = True
        self.sidebar.set_toggle_button_state(True)

    def toggle_camera(self):
        if self.camera_on:
            # Matikan
            self.stop_system()
            self.camera_view.clear()
            self.camera_view.setText("Camera Stopped")
            self.camera_on = False
        else:
            # Nyalakan
            camera_index = self.sidebar.camera_selector.currentData()
            self.start_system(camera_index)
            self.camera_on = True

        self.sidebar.set_toggle_button_state(self.camera_on)

    def stop_system(self):
        """Menghentikan kedua worker dengan aman"""
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None
        
        if self.ml_thread:
            self.ml_thread.stop()
            self.ml_thread = None

    def closeEvent(self, event):
        self.stop_system()
        event.accept()