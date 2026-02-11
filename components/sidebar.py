from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QPushButton, QComboBox
from PySide6.QtCore import Signal
from styles import SIDEBAR_STYLE

class Sidebar(QFrame):
    # Definisi sinyal kustom untuk dikirim ke MainWindow
    camera_changed = Signal(int)
    flip_requested = Signal()
    toggle_requested = Signal()

    def __init__(self, available_cameras):
        super().__init__()
        self.setMinimumWidth(200)
        self.setMaximumWidth(300)
        self.setStyleSheet(SIDEBAR_STYLE)

        self.setup_ui(available_cameras)

    def setup_ui(self, available_cameras):
        layout = QVBoxLayout(self)

        # --- Camera Info Section ---
        title = QLabel("Camera Info")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        info_layout = QVBoxLayout()
        info_layout.setContentsMargins(12, 0, 0, 0)
        info_layout.setSpacing(4)

        self.resolution_label = QLabel("Resolution: -")
        self.fps_label = QLabel("FPS: -")
        self.current_fps_label = QLabel("Current FPS: -")
        self.prediction_speed_label = QLabel("Prediction Speed: -")

        info_layout.addWidget(self.resolution_label)
        info_layout.addWidget(self.fps_label)
        info_layout.addWidget(self.current_fps_label)
        info_layout.addWidget(self.prediction_speed_label)
        layout.addLayout(info_layout)

        # --- Camera Device Section ---
        cam_label = QLabel("Camera Device")
        cam_label.setObjectName("sectionTitle")
        layout.addWidget(cam_label)

        self.camera_selector = QComboBox()
        for idx in available_cameras:
            self.camera_selector.addItem(f"Camera {idx}", idx)
        
        # Kirim sinyal saat combobox berubah
        self.camera_selector.currentIndexChanged.connect(
            lambda: self.camera_changed.emit(self.camera_selector.currentData())
        )
        layout.addWidget(self.camera_selector)

        # --- Buttons ---
        self.flip_button = QPushButton("Flip Horizontal")
        self.flip_button.clicked.connect(self.flip_requested.emit)
        layout.addWidget(self.flip_button)

        self.toggle_btn = QPushButton("Stop Camera")
        self.toggle_btn.clicked.connect(self.toggle_requested.emit)
        layout.addWidget(self.toggle_btn)

        layout.addStretch()

    # --- Fungsi Update UI dari luar ---
    def update_info(self, width, height, fps):
        self.resolution_label.setText(f"Resolution: {width} x {height}")
        self.fps_label.setText(f"FPS: {fps:.1f}")

    def update_current_fps(self, fps):
        self.current_fps_label.setText(f"Current FPS: {fps:.1f}")

    def update_prediction_speed(self, speed_ms):
        self.prediction_speed_label.setText(f"Prediction Speed: {speed_ms:.1f} ms")
    
    def set_toggle_button_state(self, is_camera_on):
        if is_camera_on:
            self.toggle_btn.setText("Stop Camera")
        else:
            self.toggle_btn.setText("Start Camera")
            self.resolution_label.setText("Resolution: -")
            self.fps_label.setText("FPS: -")
            self.current_fps_label.setText("Current FPS: -")