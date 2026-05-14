"""
components/video_player.py
===========================
Reusable video player widget.
- Video: QMediaPlayer (hardware-accelerated, smooth even for 4K 60fps).
- Folder: OpenCV + slider scrubbing (no play button, FPS unknown).
"""

import os
import cv2
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QSizePolicy, QFrame, QStackedWidget
)
from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QImage, QPixmap, QIcon
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget

from utils.resource_path import resource_path


class VideoPlayer(QWidget):
    """Video player widget with transport controls."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._source_type = None   # "video" or "folder"
        self._folder_files = []
        self._total_frames = 0
        self._fps = 30.0
        self._current_frame = 0
        self._slider_pressed = False
        self._duration_ms = 0

        self._setup_ui()
        self._setup_media_player()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Stacked display: page 0 = QVideoWidget (video), page 1 = QLabel (folder)
        self._display_stack = QStackedWidget()
        self._display_stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Page 0: Video widget (hardware-accelerated)
        self._video_widget = QVideoWidget()
        self._video_widget.setStyleSheet("background-color: #0d0d0d;")
        self._display_stack.addWidget(self._video_widget)

        # Page 1: Image label (folder scrubbing)
        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._image_label.setStyleSheet("background-color: #0d0d0d; border: none;")
        self._display_stack.addWidget(self._image_label)

        layout.addWidget(self._display_stack, 1)

        # ── Transport bar ──
        self._bar = QFrame()
        self._bar.setFixedHeight(44)
        self._bar.setStyleSheet("""
            QFrame { background-color: #1e1e1e; border-top: 1px solid #333333; }
            QLabel { font-family: 'Segoe UI', sans-serif; color: #a0a0a0; font-size: 11px; border: none; }
        """)

        bar_layout = QHBoxLayout(self._bar)
        bar_layout.setContentsMargins(10, 0, 10, 0)
        bar_layout.setSpacing(8)

        # Play/Pause button
        self._btn_play = QPushButton()
        self._btn_play.setIcon(QIcon(resource_path("assets/play.svg")))
        self._btn_play.setFixedSize(32, 32)
        self._btn_play.setStyleSheet("""
            QPushButton {
                background-color: #383838; color: #e0e0e0;
                border: 1px solid #4a4a4a; border-radius: 16px;
            }
            QPushButton:hover { background-color: #454545; }
        """)
        self._btn_play.clicked.connect(self.toggle_play)
        bar_layout.addWidget(self._btn_play)

        # Time label
        self._lbl_time = QLabel("0:00")
        self._lbl_time.setStyleSheet(
            "color: #cccccc; font-family: 'Consolas', monospace; "
            "font-size: 12px; min-width: 36px;"
        )
        bar_layout.addWidget(self._lbl_time)

        # Seek slider
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setValue(0)
        self._slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 4px; background: #333; border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #4aa3f5; width: 12px; height: 12px;
                margin: -4px 0; border-radius: 6px;
            }
            QSlider::sub-page:horizontal {
                background: #4aa3f5; border-radius: 2px;
            }
        """)
        self._slider.sliderPressed.connect(self._on_slider_pressed)
        self._slider.sliderReleased.connect(self._on_slider_released)
        self._slider.sliderMoved.connect(self._on_slider_moved)
        bar_layout.addWidget(self._slider, 1)

        # Duration label
        self._lbl_duration = QLabel("0:00")
        self._lbl_duration.setStyleSheet(
            "color: #cccccc; font-family: 'Consolas', monospace; "
            "font-size: 12px; min-width: 36px;"
        )
        bar_layout.addWidget(self._lbl_duration)

        # Frame counter
        self._lbl_frame = QLabel("")
        self._lbl_frame.setStyleSheet(
            "color: #858585; font-family: 'Consolas', monospace; font-size: 11px;"
        )
        bar_layout.addWidget(self._lbl_frame)

        layout.addWidget(self._bar)

    def _setup_media_player(self):
        """Initialize QMediaPlayer for video file playback."""
        self._audio_output = QAudioOutput()
        self._audio_output.setVolume(0.5)

        self._player = QMediaPlayer()
        self._player.setAudioOutput(self._audio_output)
        self._player.setVideoOutput(self._video_widget)

        self._player.durationChanged.connect(self._on_duration_changed)
        self._player.positionChanged.connect(self._on_position_changed)
        self._player.playbackStateChanged.connect(self._on_state_changed)

    # ──── Public API ────

    def load_video(self, path):
        """Load a video file for playback using QMediaPlayer."""
        self.stop()
        self._release_folder()

        self._source_type = "video"

        # Get frame info via OpenCV (for frame counter display)
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

            # Extract first frame as thumbnail
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame_rgb.shape
                qimg = QImage(frame_rgb.data.tobytes(), w, h, ch * w,
                              QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg)
                scaled = pixmap.scaled(
                    self._image_label.size(),
                    Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self._image_label.setPixmap(scaled)

            cap.release()
        else:
            self._total_frames = 0
            self._fps = 30.0

        # Load into QMediaPlayer
        self._player.setSource(QUrl.fromLocalFile(path))

        # Show thumbnail (page 1) until play is pressed
        self._display_stack.setCurrentIndex(1)

        # Show play button & time labels
        self._btn_play.setVisible(True)
        self._lbl_time.setVisible(True)
        self._lbl_duration.setVisible(True)

        self._current_frame = 0
        self._update_frame_label(0)

    def load_folder(self, path):
        """Load an image folder for scrubbing (no play, FPS unknown)."""
        self.stop()
        self._release_folder()

        self._source_type = "folder"
        valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        files = [f for f in os.listdir(path)
                 if os.path.splitext(f)[1].lower() in valid_ext]
        files.sort(key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
        self._folder_files = [os.path.join(path, f) for f in files]

        self._total_frames = len(self._folder_files)
        self._fps = 30.0
        self._current_frame = 0

        self._slider.setMaximum(max(0, self._total_frames - 1))
        self._slider.setValue(0)

        # Folder mode: show image label, hide play/time
        self._display_stack.setCurrentIndex(1)
        self._btn_play.setVisible(False)
        self._lbl_time.setVisible(False)
        self._lbl_duration.setVisible(False)

        self._show_folder_frame(0)

    def toggle_play(self):
        if self._source_type != "video":
            return
        if self._player.playbackState() == QMediaPlayer.PlayingState:
            self.pause()
        else:
            self.play()

    def play(self):
        if self._source_type != "video":
            return
        self._display_stack.setCurrentIndex(0)
        self._player.play()

    def pause(self):
        if self._source_type == "video":
            self._player.pause()

    def stop(self):
        if self._source_type == "video":
            self._player.stop()
        self._btn_play.setIcon(QIcon(resource_path("assets/play.svg")))
        self._current_frame = 0

    def cleanup(self):
        self.stop()
        self._player.setSource(QUrl())
        self._release_folder()

    # ──── QMediaPlayer signal handlers ────

    def _on_duration_changed(self, duration_ms):
        self._duration_ms = duration_ms
        self._slider.setMaximum(duration_ms)
        self._lbl_duration.setText(self._fmt_time(duration_ms / 1000))

    def _on_position_changed(self, position_ms):
        if not self._slider_pressed:
            self._slider.setValue(position_ms)
        self._lbl_time.setText(self._fmt_time(position_ms / 1000))

        # Estimate frame index
        if self._fps > 0:
            self._current_frame = int(position_ms / 1000 * self._fps)
            self._update_frame_label(self._current_frame)

    def _on_state_changed(self, state):
        if state == QMediaPlayer.PlayingState:
            self._btn_play.setIcon(QIcon(resource_path("assets/pause.svg")))
        else:
            self._btn_play.setIcon(QIcon(resource_path("assets/play.svg")))

    # ──── Slider callbacks ────

    def _on_slider_pressed(self):
        self._slider_pressed = True

    def _on_slider_released(self):
        self._slider_pressed = False
        value = self._slider.value()

        if self._source_type == "video":
            self._player.setPosition(value)
        elif self._source_type == "folder":
            self._show_folder_frame(value)

    def _on_slider_moved(self, value):
        if self._source_type == "video":
            self._player.setPosition(value)
            self._lbl_time.setText(self._fmt_time(value / 1000))
        elif self._source_type == "folder":
            self._show_folder_frame(value)

    # ──── Folder-specific display ────

    def _show_folder_frame(self, idx):
        idx = max(0, min(idx, self._total_frames - 1))
        self._current_frame = idx

        if 0 <= idx < len(self._folder_files):
            img = cv2.imread(self._folder_files[idx])
            if img is not None:
                frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, ch = frame_rgb.shape
                qimg = QImage(frame_rgb.data.tobytes(), w, h, ch * w,
                              QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg)
                scaled = pixmap.scaled(
                    self._image_label.size(),
                    Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self._image_label.setPixmap(scaled)

        if not self._slider_pressed:
            self._slider.setValue(idx)
        self._update_frame_label(idx)

    # ──── Helpers ────

    def _update_frame_label(self, idx):
        if self._total_frames > 0:
            self._lbl_frame.setText(f"Frame: {idx} / {self._total_frames - 1}")
        else:
            self._lbl_frame.setText("")

    def _release_folder(self):
        self._folder_files = []
        self._total_frames = 0
        self._source_type = None

    @staticmethod
    def _fmt_time(seconds):
        m, s = divmod(int(seconds), 60)
        return f"{m}:{s:02d}"
