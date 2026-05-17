import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QFileDialog, QProgressBar, QStackedWidget, QSizePolicy,
    QInputDialog
)
from PySide6.QtCore import Signal, Qt, QTimer, QSettings
from PySide6.QtGui import QPixmap, QImage, QIcon
import cv2
import time

from components.record_dialog import RecordDialog
from components.video_player import VideoPlayer
from workers.prediction_worker import PredictionWorker
from utils.resource_path import resource_path
from components.range_slider import RangeSlider


class InputPage(QWidget):
    """
    Page 1: Input & Prediction.
    Record video, import video, or import frame folder.
    Then start offline batch prediction → output CSV.
    """
    prediction_finished = Signal(str)  # csv_path

    def __init__(self):
        super().__init__()
        self.source_path = None
        self.source_type = None
        self.folder_fps = None
        self.prediction_worker = None
        self.record_widget = None
        self.video_player = None
        
        self.start_timer = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_time_elapsed)

        self.setup_ui()

    def setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ====== LEFT PANEL (Preview / Record) ======
        self.left_stack = QStackedWidget()

        # -- Page 0: Prompt --
        self.prompt_page = self._create_prompt_page()
        self.left_stack.addWidget(self.prompt_page)

        # -- Page 1: Preview --
        self.preview_page = self._create_preview_page()
        self.left_stack.addWidget(self.preview_page)

        # -- Page 2: Record --
        self.record_page_wrapper = QWidget()
        self.record_page_layout = QVBoxLayout(self.record_page_wrapper)
        self.record_page_layout.setContentsMargins(8, 8, 8, 8)
        self.left_stack.addWidget(self.record_page_wrapper)

        self.left_stack.setCurrentIndex(0)
        main_layout.addWidget(self.left_stack, 4)

        # ====== RIGHT SIDEBAR ======
        sidebar = self._create_sidebar()
        main_layout.addWidget(sidebar, 1)

    def _create_prompt_page(self):
        page = QFrame()
        page.setStyleSheet("background-color: #1a1a1a; border: none;")
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignCenter)

        icon_label = QLabel()
        icon_label.setPixmap(QIcon(resource_path("assets/play_dark.svg")).pixmap(48, 48))
        icon_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(icon_label)

        text = QLabel("Select a video source from the panel\nor record directly from camera")
        text.setStyleSheet("color: #555555; font-family: 'Segoe UI', 'Roboto', sans-serif; font-size: 14px; font-weight: 600;")
        text.setAlignment(Qt.AlignCenter)
        layout.addWidget(text)

        return page

    def _create_preview_page(self):
        page = QFrame()
        page.setStyleSheet("background-color: #1a1a1a; border: none;")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Source info header
        self.source_info_header = QLabel("")
        self.source_info_header.setStyleSheet("""
            color: #858585; font-family: 'Consolas', monospace; font-size: 11px;
            padding: 6px 12px;
            background-color: #1e1e1e;
            border: 1px solid #333333;
            border-radius: 4px;
        """)
        layout.addWidget(self.source_info_header)

        # Video player
        self.video_player = VideoPlayer()
        layout.addWidget(self.video_player, 1)

        return page

    def _create_sidebar(self):
        sidebar = QFrame()
        sidebar.setMinimumWidth(280)
        sidebar.setMaximumWidth(320)
        sidebar.setStyleSheet("""
            QFrame { background-color: #252526; border: none; border-left: 1px solid #333333; }
            QLabel { font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', sans-serif; color: #d4d4d4; font-size: 12px; }
        """)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(6)

        # Title
        title = QLabel("SOURCE INPUT")
        title.setStyleSheet("""
            font-family: 'Segoe UI', 'Roboto', sans-serif;
            font-size: 13px; font-weight: 800; color: #4aa3f5;
            letter-spacing: 1.5px; margin-bottom: 4px;
        """)
        layout.addWidget(title)

        # Description
        desc = QLabel("Select a video source to analyze")
        desc.setStyleSheet("font-family: 'Segoe UI', 'Roboto', sans-serif; color: #a0a0a0; font-size: 11px; margin-bottom: 6px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Button style
        btn_style = """
            QPushButton {
                background-color: #383838; color: #e0e0e0;
                border: 1px solid #4a4a4a; border-radius: 4px;
                padding: 8px 12px; font-size: 12px;
                font-family: 'Segoe UI', 'Roboto', sans-serif;
                text-align: left;
            }
            QPushButton:hover { background-color: #454545; }
            QPushButton:pressed { background-color: #505050; }
        """

        self.btn_record = QPushButton("  Record from Camera")
        self.btn_record.setStyleSheet(btn_style)
        self.btn_record.clicked.connect(self.open_recorder)
        layout.addWidget(self.btn_record)

        self.btn_import_video = QPushButton("  Import Video File")
        self.btn_import_video.setStyleSheet(btn_style)
        self.btn_import_video.clicked.connect(self.import_video)
        layout.addWidget(self.btn_import_video)

        self.btn_import_folder = QPushButton("  Import Frame Folder")
        self.btn_import_folder.setStyleSheet(btn_style)
        self.btn_import_folder.clicked.connect(self.import_folder)
        layout.addWidget(self.btn_import_folder)

        # Separator
        self._add_separator(layout)

        # Source info
        self._add_section_title(layout, "SOURCE INFO")

        self.lbl_source_name = QLabel("File: —")
        self.lbl_source_name.setWordWrap(True)
        layout.addWidget(self.lbl_source_name)

        self.lbl_source_detail = QLabel("Details: —")
        self.lbl_source_detail.setWordWrap(True)
        layout.addWidget(self.lbl_source_detail)

        self._add_separator(layout)

        # Frame Range
        self._add_section_title(layout, "FRAME RANGE")
        self.lbl_frame_range = QLabel("Range: All frames")
        self.lbl_frame_range.setStyleSheet("color: #a0a0a0; font-family: 'Consolas', monospace; font-size: 11px;")
        self.lbl_frame_range.setVisible(False)
        layout.addWidget(self.lbl_frame_range)

        self.range_slider = RangeSlider()
        self.range_slider.setVisible(False)
        self.range_slider.rangeChanged.connect(self._on_range_changed)
        layout.addWidget(self.range_slider)

        self._add_separator(layout)

        # Prediction section
        self._add_section_title(layout, "PREDICTION")

        self.btn_start_predict = QPushButton("  Run Prediction")
        self.btn_start_predict.setStyleSheet("""
            QPushButton {
                background-color: #0e639c; color: #ffffff;
                border: 1px solid #1177bb; border-radius: 4px;
                padding: 8px; font-size: 12px; font-weight: 600;
                font-family: 'Segoe UI', 'Roboto', sans-serif;
            }
            QPushButton:hover { background-color: #1177bb; }
            QPushButton:pressed { background-color: #094771; }
            QPushButton:disabled { background-color: #1a3a5a; color: #555566; border: 1px solid #234b6e; }
        """)
        self.btn_start_predict.setEnabled(False)
        self.btn_start_predict.clicked.connect(self.start_prediction)
        layout.addWidget(self.btn_start_predict)

        self.btn_stop_predict = QPushButton("  Cancel")
        self.btn_stop_predict.setStyleSheet("""
            QPushButton {
                background-color: #c0392b; color: #ffffff;
                border: 1px solid #d35400; border-radius: 4px;
                padding: 8px; font-size: 12px; font-weight: 600;
                font-family: 'Segoe UI', 'Roboto', sans-serif;
            }
            QPushButton:hover { background-color: #e74c3c; }
            QPushButton:pressed { background-color: #a53125; }
        """)
        self.btn_stop_predict.setVisible(False)
        self.btn_stop_predict.clicked.connect(self.stop_prediction)
        layout.addWidget(self.btn_stop_predict)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #333333;
                border-radius: 4px;
                background-color: #1e1e1e;
                text-align: center;
                color: #d4d4d4;
                font-family: 'Segoe UI', 'Roboto', sans-serif;
                font-size: 10px;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #0e639c;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.progress_bar)

        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("color: #a0a0a0; font-family: 'Consolas', monospace; font-size: 11px;")
        self.lbl_status.setWordWrap(True)
        layout.addWidget(self.lbl_status)

        self.lbl_time_elapsed = QLabel("")
        self.lbl_time_elapsed.setStyleSheet("color: #a0a0a0; font-family: 'Consolas', monospace; font-size: 11px;")
        self.lbl_time_elapsed.setVisible(False)
        layout.addWidget(self.lbl_time_elapsed)

        # Open Dashboard button
        self.btn_open_dashboard = QPushButton("  View Results in Analysis")
        self.btn_open_dashboard.setStyleSheet("""
            QPushButton {
                background-color: transparent; color: #4aa3f5;
                border: 1px solid #4aa3f5; border-radius: 4px;
                padding: 8px; font-size: 12px; font-weight: 600;
                font-family: 'Segoe UI', 'Roboto', sans-serif;
            }
            QPushButton:hover { background-color: #4aa3f5; color: #ffffff; }
        """)
        self.btn_open_dashboard.setVisible(False)
        layout.addWidget(self.btn_open_dashboard)

        layout.addStretch()
        return sidebar

    def _add_separator(self, layout):
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("background-color: #333333; max-height: 1px; margin-top: 8px; margin-bottom: 4px;")
        layout.addWidget(sep)

    def _add_section_title(self, layout, text):
        lbl = QLabel(text)
        lbl.setStyleSheet("""
            font-family: 'Segoe UI', 'Roboto', sans-serif;
            font-size: 10px; font-weight: 700; color: #858585;
            letter-spacing: 1.5px; margin-top: 4px; margin-bottom: 2px;
        """)
        layout.addWidget(lbl)

    def _on_range_changed(self, low, high):
        self.lbl_frame_range.setText(f"Range: {low} - {high}")

    # ==================
    # SOURCE ACTIONS
    # ==================

    def open_recorder(self):
        if self.record_widget:
            self.record_widget.cleanup()

        self.record_widget = RecordDialog()
        self.record_widget.recording_finished.connect(self._on_recording_done)
        self.record_widget.closed.connect(self._close_recorder)

        while self.record_page_layout.count():
            w = self.record_page_layout.takeAt(0).widget()
            if w:
                w.deleteLater()

        self.record_page_layout.addWidget(self.record_widget)
        self.left_stack.setCurrentIndex(2)

    def _on_recording_done(self, video_path):
        self.set_source(video_path, "video")

    def _close_recorder(self):
        if self.source_path:
            self.left_stack.setCurrentIndex(1)
        else:
            self.left_stack.setCurrentIndex(0)

    def import_video(self):
        settings = QSettings("AplikasiSkripsi", "MicroExpression")
        last_dir = settings.value("last_input_dir", "")
        
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Video",
            last_dir, "Video Files (*.avi *.mp4 *.mkv *.mov *.wmv);;All Files (*)"
        )
        if path:
            settings.setValue("last_input_dir", os.path.dirname(path))
            self.set_source(path, "video")

    def import_folder(self):
        settings = QSettings("AplikasiSkripsi", "MicroExpression")
        last_dir = settings.value("last_input_dir", "")
        
        path = QFileDialog.getExistingDirectory(
            self, "Import Frame Folder", last_dir
        )
        if path:
            settings.setValue("last_input_dir", path)

            # Minta pengguna memasukkan FPS asli dari frame folder
            fps_val, ok = QInputDialog.getDouble(
                self, "Input FPS",
                "Masukkan FPS asli dari frame folder ini:\n"
                "(Frame akan di-resample ke 30 FPS untuk prediksi)",
                30.0, 1.0, 240.0, 1
            )
            if not ok:
                return
            self.folder_fps = fps_val

            self.set_source(path, "folder")

    def set_source(self, path, source_type):
        self.source_path = path
        self.source_type = source_type

        name = os.path.basename(path)
        self.lbl_source_name.setText(f"File: {name}")

        if source_type == "video":
            cap = cv2.VideoCapture(path)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = total / fps if fps > 0 else 0

                self.lbl_source_detail.setText(
                    f"Resolution:  {w}x{h}\n"
                    f"FPS:              {fps:.1f}\n"
                    f"Total Frames: {total}\n"
                    f"Duration:       {duration:.1f}s"
                )
                self.source_info_header.setText(
                    f"Video: {name}  |  {w}x{h}  |  {total} frames  |  {duration:.1f}s"
                )
                cap.release()

                self.range_slider.setMinimum(0)
                self.range_slider.setMaximum(total)
                self.range_slider.setLow(0)
                self.range_slider.setHigh(total)
                self.range_slider.setVisible(True)
                self.lbl_frame_range.setText(f"Range: 0 - {total}")
                self.lbl_frame_range.setVisible(True)

                self.video_player.load_video(path)
            else:
                self.lbl_source_detail.setText("Details: Failed to open video")
        else:
            valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
            files = [f for f in os.listdir(path)
                     if os.path.splitext(f)[1].lower() in valid_ext]
            total = len(files)

            self.lbl_source_detail.setText(
                f"Type:              Frame Folder\n"
                f"FPS:               {self.folder_fps:.1f}\n"
                f"Total Frames:  {total}"
            )
            self.source_info_header.setText(
                f"Folder: {name}  |  {self.folder_fps:.1f} fps  |  {total} frame images"
            )

            self.range_slider.setMinimum(0)
            self.range_slider.setMaximum(total)
            self.range_slider.setLow(0)
            self.range_slider.setHigh(total)
            self.range_slider.setVisible(True)
            self.lbl_frame_range.setText(f"Range: 0 - {total}")
            self.lbl_frame_range.setVisible(True)

            self.video_player.load_folder(path)

        self.left_stack.setCurrentIndex(1)
        self.btn_start_predict.setEnabled(True)
        self.btn_open_dashboard.setVisible(False)
        self.lbl_status.setText("")
        self.lbl_time_elapsed.setVisible(False)



    # ==================
    # PREDICTION
    # ==================

    def start_prediction(self):
        if not self.source_path:
            return

        settings = QSettings("AplikasiSkripsi", "MicroExpression")
        last_out_dir = settings.value("last_output_dir", "")

        output_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", last_out_dir
        )
        
        if not output_dir:
            return
            
        settings.setValue("last_output_dir", output_dir)

        self.btn_start_predict.setEnabled(False)
        self.btn_stop_predict.setVisible(True)
        self.btn_open_dashboard.setVisible(False)
        self.btn_record.setEnabled(False)
        self.btn_import_video.setEnabled(False)
        self.btn_import_folder.setEnabled(False)
        self.range_slider.setEnabled(False)

        # Pause player untuk melepas file handle
        if self.video_player:
            self.video_player.pause()

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        self.lbl_time_elapsed.setVisible(True)
        self.lbl_time_elapsed.setText("Time Elapsed: 00:00")
        self.start_timer = time.time()
        self.timer.start(1000)

        low, high = self.range_slider.getRange()

        self.prediction_worker = PredictionWorker(
            self.source_path, output_dir=output_dir, folder_fps=self.folder_fps,
            start_frame=low, end_frame=high
        )
        self.prediction_worker.progress.connect(self._on_progress)
        self.prediction_worker.status.connect(self._on_status)
        self.prediction_worker.finished.connect(self._on_prediction_done)
        self.prediction_worker.error.connect(self._on_prediction_error)
        self.prediction_worker.start()

    def stop_prediction(self):
        if self.prediction_worker:
            self.prediction_worker.stop()
            self.prediction_worker.wait()
            self.prediction_worker = None

        self._reset_predict_ui()
        self.lbl_status.setText("Prediction cancelled")

    def _on_progress(self, current, total):
        if total > 0:
            pct = int((current / total) * 100)
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(current)
            self.progress_bar.setFormat(f"  {current}/{total}  ({pct}%)")

    def _on_status(self, text):
        self.lbl_status.setText(text)

    def _on_prediction_done(self, csv_path):
        self._reset_predict_ui()
        self.lbl_status.setText(f"Saved: {os.path.basename(csv_path)}")
        self.last_csv_path = csv_path

        self.btn_open_dashboard.setVisible(True)
        self.prediction_finished.emit(csv_path)

    def _on_prediction_error(self, msg):
        self._reset_predict_ui()
        self.lbl_status.setText(f"Error: {msg}")

    def _update_time_elapsed(self):
        elapsed = int(time.time() - self.start_timer)
        mins, secs = divmod(elapsed, 60)
        hrs, mins = divmod(mins, 60)
        if hrs > 0:
            self.lbl_time_elapsed.setText(f"Time Elapsed: {hrs:02d}:{mins:02d}:{secs:02d}")
        else:
            self.lbl_time_elapsed.setText(f"Time Elapsed: {mins:02d}:{secs:02d}")

    def _reset_predict_ui(self):
        if self.timer.isActive():
            self.timer.stop()
        self.btn_start_predict.setEnabled(True)
        self.btn_stop_predict.setVisible(False)
        self.btn_record.setEnabled(True)
        self.btn_import_video.setEnabled(True)
        self.btn_import_folder.setEnabled(True)
        self.range_slider.setEnabled(True)
        self.progress_bar.setVisible(False)

    def cleanup(self):
        if self.prediction_worker:
            self.prediction_worker.stop()
            self.prediction_worker.wait()
        if self.record_widget:
            self.record_widget.cleanup()
        if self.video_player:
            self.video_player.cleanup()
