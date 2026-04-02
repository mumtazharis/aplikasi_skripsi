import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QFileDialog, QProgressBar, QStackedWidget, QSizePolicy
)
from PySide6.QtCore import Signal, Qt, QTimer
from PySide6.QtGui import QPixmap, QImage
import cv2
import time

from components.record_dialog import RecordDialog
from workers.prediction_worker import PredictionWorker


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
        self.prediction_worker = None
        self.record_widget = None
        
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

        icon_label = QLabel("▶")
        icon_label.setStyleSheet("font-size: 48px; color: #3a3a3a;")
        icon_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(icon_label)

        text = QLabel("Select a video source from the panel\nor record directly from camera")
        text.setStyleSheet("color: #666; font-size: 13px;")
        text.setAlignment(Qt.AlignCenter)
        layout.addWidget(text)

        return page

    def _create_preview_page(self):
        page = QFrame()
        page.setStyleSheet("background-color: #1a1a1a; border: none;")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 12)

        # Source info header
        self.source_info_header = QLabel("")
        self.source_info_header.setStyleSheet("""
            color: #999; font-size: 11px;
            padding: 6px 12px;
            background-color: #232323;
            border: 1px solid #3a3a3a;
            border-radius: 3px;
        """)
        layout.addWidget(self.source_info_header)

        # Thumbnail preview
        self.preview_thumb = QLabel("Preview")
        self.preview_thumb.setAlignment(Qt.AlignCenter)
        self.preview_thumb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview_thumb.setStyleSheet("""
            background-color: #1e1e1e;
            color: #555;
            font-size: 13px;
            border: 1px solid #3a3a3a;
            border-radius: 3px;
        """)
        layout.addWidget(self.preview_thumb, 1)

        return page

    def _create_sidebar(self):
        sidebar = QFrame()
        sidebar.setMinimumWidth(280)
        sidebar.setMaximumWidth(320)
        sidebar.setStyleSheet("""
            QFrame { background-color: #2d2d2d; border: none; border-left: 1px solid #3a3a3a; }
            QLabel { color: #ccc; font-size: 12px; }
        """)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(6)

        # Title
        title = QLabel("SOURCE INPUT")
        title.setStyleSheet("""
            font-size: 11px; font-weight: bold; color: #2d8ceb;
            letter-spacing: 2px;
        """)
        layout.addWidget(title)

        # Description
        desc = QLabel("Select a video source to analyze")
        desc.setStyleSheet("color: #777; font-size: 11px; margin-bottom: 6px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Button style — Adobe flat dark
        btn_style = """
            QPushButton {
                background-color: #383838; color: #e0e0e0;
                border: 1px solid #4a4a4a; border-radius: 3px;
                padding: 10px 12px; font-size: 12px;
                text-align: left;
            }
            QPushButton:hover { background-color: #454545; border-color: #555; }
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

        # Prediction section
        self._add_section_title(layout, "PREDICTION")

        self.btn_start_predict = QPushButton("  Run Prediction")
        self.btn_start_predict.setStyleSheet("""
            QPushButton {
                background-color: #2d8ceb; color: white;
                border: none; border-radius: 3px;
                padding: 12px; font-size: 12px; font-weight: bold;
            }
            QPushButton:hover { background-color: #4aa3f5; }
            QPushButton:disabled { background-color: #1a3a5a; color: #556; }
        """)
        self.btn_start_predict.setEnabled(False)
        self.btn_start_predict.clicked.connect(self.start_prediction)
        layout.addWidget(self.btn_start_predict)

        self.btn_stop_predict = QPushButton("  Cancel")
        self.btn_stop_predict.setStyleSheet("""
            QPushButton {
                background-color: #c0392b; color: white;
                border: none; border-radius: 3px;
                padding: 8px; font-size: 11px;
            }
            QPushButton:hover { background-color: #e04838; }
        """)
        self.btn_stop_predict.setVisible(False)
        self.btn_stop_predict.clicked.connect(self.stop_prediction)
        layout.addWidget(self.btn_stop_predict)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #3a3a3a;
                border-radius: 3px;
                background-color: #232323;
                text-align: center;
                color: #ccc;
                font-size: 10px;
                height: 20px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1a5a9e, stop:1 #2d8ceb);
                border-radius: 2px;
            }
        """)
        layout.addWidget(self.progress_bar)

        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("color: #888; font-size: 10px;")
        self.lbl_status.setWordWrap(True)
        layout.addWidget(self.lbl_status)

        self.lbl_time_elapsed = QLabel("")
        self.lbl_time_elapsed.setStyleSheet("color: #888; font-size: 10px; font-weight: bold;")
        self.lbl_time_elapsed.setVisible(False)
        layout.addWidget(self.lbl_time_elapsed)

        # Open Dashboard button
        self.btn_open_dashboard = QPushButton("  View Results in Analysis")
        self.btn_open_dashboard.setStyleSheet("""
            QPushButton {
                background-color: #383838; color: #2d8ceb;
                border: 1px solid #2d8ceb; border-radius: 3px;
                padding: 10px; font-size: 12px; font-weight: bold;
            }
            QPushButton:hover { background-color: #2d8ceb; color: white; }
        """)
        self.btn_open_dashboard.setVisible(False)
        layout.addWidget(self.btn_open_dashboard)

        layout.addStretch()
        return sidebar

    def _add_separator(self, layout):
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("background-color: #3a3a3a; max-height: 1px; margin-top: 4px; margin-bottom: 2px;")
        layout.addWidget(sep)

    def _add_section_title(self, layout, text):
        lbl = QLabel(text)
        lbl.setStyleSheet("""
            font-size: 10px; font-weight: bold; color: #999;
            letter-spacing: 2px; margin-top: 2px;
        """)
        layout.addWidget(lbl)

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
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Video",
            "", "Video Files (*.avi *.mp4 *.mkv *.mov *.wmv);;All Files (*)"
        )
        if path:
            self.set_source(path, "video")

    def import_folder(self):
        path = QFileDialog.getExistingDirectory(
            self, "Import Frame Folder", ""
        )
        if path:
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

                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self._show_thumbnail(frame_rgb)

                cap.release()
            else:
                self.lbl_source_detail.setText("Details: Failed to open video")
        else:
            valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
            files = [f for f in os.listdir(path)
                     if os.path.splitext(f)[1].lower() in valid_ext]
            total = len(files)

            self.lbl_source_detail.setText(
                f"Type:              Frame Folder\n"
                f"Total Frames:  {total}"
            )
            self.source_info_header.setText(
                f"Folder: {name}  |  {total} frame images"
            )

            if files:
                files.sort()
                first = os.path.join(path, files[0])
                img = cv2.imread(first)
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    self._show_thumbnail(img_rgb)

        self.left_stack.setCurrentIndex(1)
        self.btn_start_predict.setEnabled(True)
        self.btn_open_dashboard.setVisible(False)
        self.lbl_status.setText("")
        self.lbl_time_elapsed.setVisible(False)

    def _show_thumbnail(self, frame_rgb):
        h, w, ch = frame_rgb.shape
        bpl = ch * w
        image = QImage(frame_rgb.data, w, h, bpl, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)

        scaled = pixmap.scaled(
            self.preview_thumb.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.preview_thumb.setPixmap(scaled)

    # ==================
    # PREDICTION
    # ==================

    def start_prediction(self):
        if not self.source_path:
            return

        output_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", ""
        )
        
        if not output_dir:
            return

        self.btn_start_predict.setEnabled(False)
        self.btn_stop_predict.setVisible(True)
        self.btn_open_dashboard.setVisible(False)
        self.btn_record.setEnabled(False)
        self.btn_import_video.setEnabled(False)
        self.btn_import_folder.setEnabled(False)

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        self.lbl_time_elapsed.setVisible(True)
        self.lbl_time_elapsed.setText("Time Elapsed: 00:00")
        self.start_timer = time.time()
        self.timer.start(1000)

        self.prediction_worker = PredictionWorker(self.source_path, output_dir=output_dir)
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
        self.progress_bar.setVisible(False)

    def cleanup(self):
        if self.prediction_worker:
            self.prediction_worker.stop()
            self.prediction_worker.wait()
        if self.record_widget:
            self.record_widget.cleanup()
