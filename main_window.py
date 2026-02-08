import cv2
from PySide6.QtWidgets import QWidget, QHBoxLayout
from PySide6.QtGui import QImage

from camera_worker import CameraWorker
from components.camera_view import CameraView
from components.sidebar import Sidebar

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Camera")
        self.resize(1000, 600)

        self.camera_on = True
        self.camera_thread = None
        self.flip_horizontal = False

        # Cek kamera yang tersedia
        self.available_cameras = self.list_cameras()

        # Inisialisasi Komponen UI
        self.camera_view = CameraView()
        self.sidebar = Sidebar(self.available_cameras)

        # Susun Layout
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addWidget(self.camera_view, 4)
        main_layout.addWidget(self.sidebar, 1)

        # Hubungkan Sinyal Sidebar ke Logika MainWindow
        self.sidebar.camera_changed.connect(self.change_camera)
        self.sidebar.flip_requested.connect(self.toggle_flip)
        self.sidebar.toggle_requested.connect(self.toggle_camera)

        # Mulai Kamera Default
        default_camera = self.available_cameras[0] if self.available_cameras else 0
        self.start_camera_thread(default_camera)

    def list_cameras(self, max_tested=5):
        available = []
        for i in range(max_tested):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(i)
                cap.release()
        return available

    def start_camera_thread(self, camera_index):
        self.camera_thread = CameraWorker(camera_index)
        
        # Connect signals dari CameraWorker ke UI
        self.camera_thread.frame_ready.connect(self.process_frame)
        self.camera_thread.camera_info.connect(self.sidebar.update_info)
        self.camera_thread.current_fps.connect(self.sidebar.update_current_fps)
        
        self.camera_thread.set_flip(self.flip_horizontal)
        self.camera_thread.start()

    def process_frame(self, frame):
        """Ubah frame OpenCV ke QImage lalu kirim ke CameraView"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        self.camera_view.update_frame(image)

    def toggle_flip(self):
        self.flip_horizontal = not self.flip_horizontal
        if self.camera_thread:
            self.camera_thread.set_flip(self.flip_horizontal)

    def change_camera(self, camera_index):
        if not self.camera_on:
            return
        if self.camera_thread:
            self.camera_thread.stop()
        self.start_camera_thread(camera_index)

    def toggle_camera(self):
        if self.camera_on:
            # Matikan kamera
            if self.camera_thread:
                self.camera_thread.stop()
                self.camera_thread = None
            self.camera_view.clear()
            self.camera_view.setText("Camera Stopped")
            self.camera_on = False
        else:
            # Nyalakan kamera
            # Ambil index kamera dari UI sidebar secara langsung atau simpan state-nya
            camera_index = self.sidebar.camera_selector.currentData()
            self.start_camera_thread(camera_index)
            self.camera_on = True

        # Update teks tombol di sidebar
        self.sidebar.set_toggle_button_state(self.camera_on)

    def closeEvent(self, event):
        if self.camera_thread is not None:
            self.camera_thread.stop()
        event.accept()