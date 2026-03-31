import cv2
import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QFrame, QSizePolicy
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QImage, QPixmap
import numpy as np
from camera_worker import CameraWorker


class RecordDialog(QWidget):
    """
    Sub-panel for recording video from camera.
    Live preview + Start/Stop recording controls.
    """
    recording_finished = Signal(str)
    closed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.camera_thread = None
        self.video_writer = None
        self.is_recording = False
        self.output_path = None
        self.frame_size = None
        self.flip_horizontal = False

        self.setup_ui()
        self.available_cameras = self._list_cameras()
        self._populate_cameras()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Title
        title = QLabel("CAMERA RECORDER")
        title.setStyleSheet("""
            font-size: 12px; font-weight: bold; color: #e0e0e0;
            letter-spacing: 2px;
            padding: 10px 14px;
            background-color: #232323;
            border: 1px solid #3a3a3a;
            border-radius: 3px;
        """)
        layout.addWidget(title)

        # Main content
        content_layout = QHBoxLayout()
        content_layout.setSpacing(10)

        # --- Preview Area ---
        self.preview_label = QLabel("Camera inactive")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(480, 360)
        self.preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview_label.setStyleSheet("""
            background-color: #1a1a1a;
            color: #555;
            font-size: 13px;
            border: 1px solid #3a3a3a;
            border-radius: 3px;
        """)
        content_layout.addWidget(self.preview_label, 3)

        # --- Controls Panel ---
        controls = QFrame()
        controls.setStyleSheet("""
            QFrame { background-color: #2d2d2d; border-radius: 3px; border: 1px solid #3a3a3a; }
            QLabel { color: #ccc; font-size: 11px; }
            QPushButton {
                background-color: #383838; color: #e0e0e0;
                border: 1px solid #4a4a4a; border-radius: 3px;
                padding: 8px; font-size: 12px;
            }
            QPushButton:hover { background-color: #454545; }
            QPushButton:disabled { color: #555; background-color: #2a2a2a; }
            QComboBox {
                background-color: #383838; color: #e0e0e0;
                border: 1px solid #4a4a4a; border-radius: 3px; padding: 6px;
            }
        """)
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(14, 14, 14, 14)
        controls_layout.setSpacing(8)

        # Camera selector
        controls_layout.addWidget(QLabel("Camera:"))
        self.camera_combo = QComboBox()
        controls_layout.addWidget(self.camera_combo)

        # Preview button
        self.preview_btn = QPushButton("Start Preview")
        self.preview_btn.clicked.connect(self.toggle_preview)
        controls_layout.addWidget(self.preview_btn)

        # Flip button
        self.flip_btn = QPushButton("Flip Horizontal")
        self.flip_btn.clicked.connect(self.toggle_flip)
        self.flip_btn.setEnabled(False)
        controls_layout.addWidget(self.flip_btn)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("background-color: #3a3a3a;")
        controls_layout.addWidget(sep)

        # Info label
        self.info_label = QLabel("Status: Idle")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("color: #888; font-size: 10px;")
        controls_layout.addWidget(self.info_label)

        # Record button
        self.record_btn = QPushButton("Start Recording")
        self.record_btn.setStyleSheet("""
            QPushButton {
                background-color: #c0392b; color: white;
                font-weight: bold; border-radius: 3px;
                padding: 10px; font-size: 12px; border: none;
            }
            QPushButton:hover { background-color: #e04838; }
            QPushButton:disabled { background-color: #4a2222; color: #888; }
        """)
        self.record_btn.setEnabled(False)
        self.record_btn.clicked.connect(self.toggle_recording)
        controls_layout.addWidget(self.record_btn)

        # Close button
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self._on_close)
        controls_layout.addWidget(self.close_btn)

        controls_layout.addStretch()
        content_layout.addWidget(controls, 1)

        layout.addLayout(content_layout)

    def _list_cameras(self, max_tested=5):
        available = []
        for i in range(max_tested):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(i)
                cap.release()
        return available

    def _populate_cameras(self):
        self.camera_combo.clear()
        for idx in self.available_cameras:
            self.camera_combo.addItem(f"Camera {idx}", idx)

    def toggle_preview(self):
        if self.camera_thread and self.camera_thread.isRunning():
            self.stop_preview()
            self.preview_btn.setText("Start Preview")
            self.record_btn.setEnabled(False)
            self.flip_btn.setEnabled(False)
        else:
            self.start_preview()
            self.preview_btn.setText("Stop Preview")
            self.record_btn.setEnabled(True)
            self.flip_btn.setEnabled(True)

    def start_preview(self):
        camera_idx = self.camera_combo.currentData()
        if camera_idx is None:
            camera_idx = 0

        self.camera_thread = CameraWorker(camera_idx)
        self.camera_thread.set_flip(self.flip_horizontal)
        self.camera_thread.frame_ready.connect(self._on_frame)
        self.camera_thread.camera_info.connect(self._on_camera_info)
        self.camera_thread.start()
        self.info_label.setText("Status: Preview active")

    def stop_preview(self):
        if self.is_recording:
            self._stop_recording()

        if self.camera_thread:
            try:
                self.camera_thread.frame_ready.disconnect()
                self.camera_thread.camera_info.disconnect()
            except:
                pass
            self.camera_thread.stop()
            self.camera_thread = None

        self.preview_label.setText("Camera inactive")
        self.preview_label.setPixmap(QPixmap())
        self.info_label.setText("Status: Preview stopped")

    def toggle_flip(self):
        self.flip_horizontal = not self.flip_horizontal
        if self.camera_thread:
            self.camera_thread.set_flip(self.flip_horizontal)

    def toggle_recording(self):
        if self.is_recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        os.makedirs("recordings", exist_ok=True)
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.output_path = os.path.join("recordings", f"record_{timestamp}.avi")

        if self.frame_size is None:
            self.frame_size = (640, 480)

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.video_writer = cv2.VideoWriter(
            self.output_path, fourcc, 30.0, self.frame_size
        )

        self.is_recording = True
        self.record_btn.setText("Stop Recording")
        self.record_btn.setStyleSheet("""
            QPushButton {
                background-color: #e6a817; color: #1a1a1a;
                font-weight: bold; border-radius: 3px;
                padding: 10px; font-size: 12px; border: none;
            }
            QPushButton:hover { background-color: #f0bd2f; }
        """)
        self.preview_btn.setEnabled(False)
        self.info_label.setText(f"Recording to: {self.output_path}")

    def _stop_recording(self):
        self.is_recording = False
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

        self.record_btn.setText("Start Recording")
        self.record_btn.setStyleSheet("""
            QPushButton {
                background-color: #c0392b; color: white;
                font-weight: bold; border-radius: 3px;
                padding: 10px; font-size: 12px; border: none;
            }
            QPushButton:hover { background-color: #e04838; }
            QPushButton:disabled { background-color: #4a2222; color: #888; }
        """)
        self.preview_btn.setEnabled(True)
        self.info_label.setText(f"Saved: {self.output_path}")

        saved_path = self.output_path

        # MATIKAN KAMERA SECARA OTOMATIS SETELAH REKAMAN SELESAI
        self.stop_preview()

        # Emit sinyal setelah kamera benar-benar mati
        if saved_path and os.path.exists(saved_path):
            self.recording_finished.emit(saved_path)

    def _on_frame(self, frame_rgb):
        if self.is_recording and self.video_writer:
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            self.video_writer.write(frame_bgr)

        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)

        scaled = pixmap.scaled(
            self.preview_label.size(),
            Qt.KeepAspectRatio,
            Qt.FastTransformation # <-- Ubah ke FastTransformation
        )
        self.preview_label.setPixmap(scaled)

    def _on_camera_info(self, width, height, fps):
        self.frame_size = (width, height)
        self.info_label.setText(f"Status: Preview active ({width}x{height} @ {fps:.0f}fps)")

    def _on_close(self):
        self.stop_preview()
        self.closed.emit()

    def cleanup(self):
        self.stop_preview()
