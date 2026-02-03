from PySide6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QFrame, QPushButton, QComboBox,  QSizePolicy
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
import cv2

from camera_worker import CameraWorker

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live Camera + Sidebar")
        self.resize(1000, 600)

        self.camera_on = True
        self.current_pixmap = None

        # ===== Kamera View =====
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setStyleSheet("background-color: black;")
        self.camera_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.camera_label.setMinimumSize(600, 450)

        # ===== Sidebar =====
        sidebar = QFrame()
        sidebar.setMinimumWidth(200)
        sidebar.setMaximumWidth(300)
        sidebar.setStyleSheet("""
            QFrame {
                background-color: #2b2b2b;
            }
            QLabel {
                color: white;
            }
        """)

        sidebar_layout = QVBoxLayout(sidebar)

        self.camera_toggle_btn = QPushButton("Stop Camera")
        self.camera_toggle_btn.clicked.connect(self.toggle_camera)
        sidebar_layout.addWidget(self.camera_toggle_btn)

        sidebar_layout.addWidget(QLabel("📷 Camera Info"))
        sidebar_layout.addWidget(QLabel("Resolution: -"))
        sidebar_layout.addWidget(QLabel("FPS: -"))

        # Label
        camera_label = QLabel("Camera Device")
        sidebar_layout.addWidget(camera_label)

        # Dropdown
       
        self.camera_selector = QComboBox()
        sidebar_layout.addWidget(self.camera_selector)
        self.available_cameras = self.list_cameras()

        for idx in self.available_cameras:
            self.camera_selector.addItem(f"Camera {idx}", idx)

        self.camera_selector.currentIndexChanged.connect(self.change_camera)


        # Tombol FLip
        self.flip_horizontal = False
        self.flip_button = QPushButton("Flip Horizontal")
        self.flip_button.clicked.connect(self.toggle_flip)

        sidebar_layout.addWidget(self.flip_button)
        
        sidebar_layout.addStretch()

        # ===== Main Layout =====
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)  # kiri, atas, kanan, bawah
        main_layout.setSpacing(0)

        main_layout.addWidget(self.camera_label, 4)
        main_layout.addWidget(sidebar, 1)


        # ===== Camera Thread =====
        default_camera = self.available_cameras[0] if self.available_cameras else 0

        self.camera_thread = CameraWorker(default_camera)
        self.camera_thread.frame_ready.connect(self.update_frame)
        self.camera_thread.start()



    def update_frame(self, frame):
        if self.flip_horizontal:
            frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w

        image = QImage(
            rgb.data, w, h, bytes_per_line, QImage.Format_RGB888
        )

        self.current_pixmap = QPixmap.fromImage(image)
        self.update_scaled_pixmap()

    def update_scaled_pixmap(self):
        if not self.current_pixmap:
            return

        scaled = self.current_pixmap.scaled(
            self.camera_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.camera_label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_scaled_pixmap()

    def toggle_flip(self):
        self.flip_horizontal = not self.flip_horizontal
    
    def list_cameras(self, max_tested=5):
        available = []
        for i in range(max_tested):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(i)
                cap.release()
        return available

    def change_camera(self):
        if not self.camera_on:
            return
        camera_index = self.camera_selector.currentData()

        # Stop kamera lama
        if self.camera_thread:
            self.camera_thread.stop()

        # Start kamera baru
        self.camera_thread = CameraWorker(camera_index)
        self.camera_thread.frame_ready.connect(self.update_frame)
        self.camera_thread.start()

    def toggle_camera(self):
        if self.camera_on:
            # Matikan kamera
            if self.camera_thread:
                self.camera_thread.stop()
                self.camera_thread = None

            self.camera_label.clear()
            self.camera_label.setText("Stop Camera")
            self.camera_toggle_btn.setText("Start Camera")
            self.camera_on = False

        else:
            # Nyalakan kamera
            camera_index = self.camera_selector.currentData()

            self.camera_thread = CameraWorker(camera_index)
            self.camera_thread.frame_ready.connect(self.update_frame)
            self.camera_thread.start()

            self.camera_toggle_btn.setText("Stop Camera")
            self.camera_on = True

