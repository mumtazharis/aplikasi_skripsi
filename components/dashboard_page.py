import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QFileDialog, QSlider, QComboBox, QSizePolicy, QScrollArea
)
from PySide6.QtCore import Qt, Signal, QSettings
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor
import cv2

from components.roi_linechart import RoiLineChartWidget
from components.playback_chart import PlaybackChart
from workers.playback_worker import PlaybackWorker, load_csv_data, load_meta


class DashboardPage(QWidget):
    """
    Page 2: Analysis Dashboard.
    Load CSV prediction results + replay video with sync.
    """

    def __init__(self):
        super().__init__()
        self.csv_data = []
        self.csv_path = None
        self.playback_worker = None
        self.total_frames = 0
        self.is_playing = False

        self.setup_ui()

    def setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ====== LEFT PANEL (Video + Chart) ======
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        # Video display
        self.video_label = QLabel("Load a CSV file to begin")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setStyleSheet("""
            background-color: #1a1a1a;
            color: #555555;
            font-family: 'Segoe UI', 'Roboto', sans-serif;
            font-size: 14px;
            font-weight: 600;
        """)
        left_layout.addWidget(self.video_label, 3)

        # Playback controls bar
        controls_bar = self._create_controls_bar()
        left_layout.addWidget(controls_bar)

        # Timeline chart
        chart_container = QFrame()
        chart_container.setStyleSheet("background-color: #232323; border: none;")
        chart_layout = QVBoxLayout(chart_container)
        chart_layout.setContentsMargins(0, 0, 0, 0)
        chart_layout.setSpacing(0)

        chart_header = QFrame()
        chart_header.setFixedHeight(24)
        chart_header.setStyleSheet("background-color: #1e1e1e; border-bottom: 1px solid #333333;")
        ch_layout = QHBoxLayout(chart_header)
        ch_layout.setContentsMargins(12, 0, 12, 0)
        ch_title = QLabel("EMOTION TIMELINE")
        ch_title.setStyleSheet("font-family: 'Segoe UI', sans-serif; color: #858585; font-size: 10px; font-weight: 700; letter-spacing: 2px;")
        ch_layout.addWidget(ch_title)
        chart_layout.addWidget(chart_header)

        self.playback_chart = PlaybackChart()
        self.playback_chart.seek_requested.connect(self._on_chart_seek)

        # ====== BAGIAN YANG DISESUAIKAN UNTUK SCROLL ======
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True) 
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn) 
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  
        self.scroll_area.setFrameShape(QFrame.NoFrame) 
        
        # 1. Masukkan chart ke dalam scroll area
        self.scroll_area.setWidget(self.playback_chart)
        
        # 2. Tambahkan scroll area ke layout
        chart_layout.addWidget(self.scroll_area)
        # ===================================================

        left_layout.addWidget(chart_container, 1)

        main_layout.addWidget(left_panel, 4)

        # ====== RIGHT SIDEBAR ======
        sidebar = self._create_sidebar()
        main_layout.addWidget(sidebar, 1)

    def _create_controls_bar(self):
        bar = QFrame()
        bar.setFixedHeight(44)
        bar.setStyleSheet("""
            QFrame { background-color: #1e1e1e; border-top: 1px solid #333333; }
            QLabel { font-family: 'Segoe UI', sans-serif; color: #a0a0a0; font-size: 11px; }
        """)

        layout = QHBoxLayout(bar)
        layout.setContentsMargins(10, 0, 10, 0)
        layout.setSpacing(8)

        # Play/Pause
        self.btn_play = QPushButton("▶")
        self.btn_play.setFixedSize(32, 32)
        self.btn_play.setStyleSheet("""
            QPushButton {
                background-color: #383838; color: #e0e0e0;
                border: 1px solid #4a4a4a; border-radius: 16px;
                font-size: 12px;
            }
            QPushButton:hover { background-color: #454545; }
            QPushButton:disabled { color: #444444; border: 1px solid #333333; }
        """)
        self.btn_play.setEnabled(False)
        self.btn_play.clicked.connect(self.toggle_play)
        layout.addWidget(self.btn_play)

        # Stop
        self.btn_stop = QPushButton("■")
        self.btn_stop.setFixedSize(32, 32)
        self.btn_stop.setStyleSheet("""
            QPushButton {
                background-color: #383838; color: #e0e0e0;
                border: 1px solid #4a4a4a; border-radius: 16px;
                font-size: 10px;
            }
            QPushButton:hover { background-color: #454545; }
            QPushButton:disabled { color: #444444; border: 1px solid #333333; }
        """)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_playback)
        layout.addWidget(self.btn_stop)

        # Current time
        self.lbl_time = QLabel("0:00")
        self.lbl_time.setStyleSheet("color: #cccccc; font-family: 'Consolas', monospace; font-size: 12px; min-width: 36px;")
        layout.addWidget(self.lbl_time)

        # Seek slider
        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setEnabled(False)
        self.seek_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: none; height: 4px; background: #383838; border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #0e639c; width: 12px; height: 12px; margin: -4px 0; border-radius: 6px;
            }
            QSlider::handle:horizontal:hover {
                background: #1177bb;
            }
            QSlider::sub-page:horizontal {
                background: #0e639c; border-radius: 2px;
            }
        """)
        self.seek_slider.valueChanged.connect(self._on_slider_seek)
        layout.addWidget(self.seek_slider, 1)

        # Total time
        self.lbl_total_time = QLabel("0:00")
        self.lbl_total_time.setStyleSheet("color: #858585; font-family: 'Consolas', monospace; font-size: 12px; min-width: 36px;")
        layout.addWidget(self.lbl_total_time)

        # Speed selector
        speed_label = QLabel("Speed:")
        layout.addWidget(speed_label)

        self.speed_combo = QComboBox()
        self.speed_combo.setStyleSheet("""
            QComboBox {
                background-color: #2d2d2d; color: #cccccc;
                border: 1px solid #4a4a4a; border-radius: 3px;
                padding: 3px 6px; font-size: 11px; font-family: 'Segoe UI', sans-serif;
            }
            QComboBox QAbstractItemView {
                background-color: #2d2d2d; color: #cccccc;
                selection-background-color: #0e639c;
            }
        """)
        self.speed_combo.addItem("0.25x", 0.25)
        self.speed_combo.addItem("0.5x", 0.5)
        self.speed_combo.addItem("1x", 1.0)
        self.speed_combo.addItem("1.5x", 1.5)
        self.speed_combo.addItem("2x", 2.0)
        self.speed_combo.addItem("4x", 4.0)
        self.speed_combo.setCurrentIndex(2)
        self.speed_combo.currentIndexChanged.connect(self._on_speed_changed)
        layout.addWidget(self.speed_combo)

        return bar

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
        title = QLabel("ANALYSIS DASHBOARD")
        title.setStyleSheet("""
            font-family: 'Segoe UI', 'Roboto', sans-serif;
            font-size: 13px; font-weight: 800; color: #4aa3f5;
            letter-spacing: 1.5px; margin-bottom: 4px;
        """)
        layout.addWidget(title)

        # Load CSV button
        self.btn_load_csv = QPushButton(" Load Analysis File")
        self.btn_load_csv.setStyleSheet("""
            QPushButton {
                background-color: #0e639c; color: #ffffff;
                border: 1px solid #1177bb; border-radius: 4px;
                padding: 8px; font-size: 12px; font-weight: 600;
                font-family: 'Segoe UI', 'Roboto', sans-serif;
            }
            QPushButton:hover { background-color: #1177bb; }
            QPushButton:pressed { background-color: #094771; }
        """)
        self.btn_load_csv.clicked.connect(self.load_csv)
        layout.addWidget(self.btn_load_csv)

        # --- Info section ---
        self._add_separator(layout)
        self._add_section_title(layout, "DATA INFO")

        self.lbl_csv_name = QLabel("File: —")
        self.lbl_csv_name.setWordWrap(True)
        layout.addWidget(self.lbl_csv_name)

        self.lbl_total_frames = QLabel("Total Frames: —")
        layout.addWidget(self.lbl_total_frames)

        self.lbl_source_ref = QLabel("Source: —")
        self.lbl_source_ref.setWordWrap(True)
        layout.addWidget(self.lbl_source_ref)
        
        # --- Current Prediction ---
        self._add_separator(layout)
        self._add_section_title(layout, "CURRENT FRAME")

        self.lbl_frame_num = QLabel("Frame: —")
        self.lbl_frame_num.setStyleSheet("font-family: 'Consolas', monospace; font-size: 12px; color: #a0a0a0; margin-bottom: 4px;")
        layout.addWidget(self.lbl_frame_num)

        # Layout horizontal 2 kolom
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(10) 

        # --- Macro card ---
        macro_card = QFrame()
        macro_card.setObjectName("macroCard")
        macro_card.setFixedHeight(80) 
        macro_card.setStyleSheet("""
            QFrame#macroCard {          
                background-color: #1e1e1e;
                border: 1px solid #333333;
                border-radius: 6px;
            }
        """)
        mc_layout = QVBoxLayout(macro_card)
        mc_layout.setContentsMargins(6, 8, 6, 8) 
        mc_layout.setSpacing(2)

        mc_header = QLabel("MACRO")
        mc_header.setStyleSheet("color: #858585; font-size: 10px; font-weight: 700; letter-spacing: 1px;")
        mc_header.setAlignment(Qt.AlignCenter)
        mc_layout.addWidget(mc_header)

        self.lbl_macro_label = QLabel("—")
        self.lbl_macro_label.setStyleSheet("font-size: 20px; font-weight: 800; color: #888888;")
        self.lbl_macro_label.setAlignment(Qt.AlignCenter)
        mc_layout.addWidget(self.lbl_macro_label)

        self.lbl_macro_conf = QLabel("Conf: —") 
        self.lbl_macro_conf.setStyleSheet("font-family: 'Consolas', monospace; font-size: 11px; color: #9e9e9e;")
        self.lbl_macro_conf.setAlignment(Qt.AlignCenter)
        mc_layout.addWidget(self.lbl_macro_conf)

        cards_layout.addWidget(macro_card)

        # --- Micro card ---
        micro_card = QFrame()
        micro_card.setObjectName("microCard")
        micro_card.setFixedHeight(80) 
        micro_card.setStyleSheet("""
            QFrame#microCard {
                background-color: #1e1e1e;
                border: 1px solid #333333;
                border-radius: 6px;
            }
        """)
        mi_layout = QVBoxLayout(micro_card)
        mi_layout.setContentsMargins(6, 8, 6, 8) 
        mi_layout.setSpacing(2)

        mi_header = QLabel("MICRO") 
        mi_header.setStyleSheet("color: #858585; font-size: 10px; font-weight: 700; letter-spacing: 1px;")
        mi_header.setAlignment(Qt.AlignCenter)
        mi_layout.addWidget(mi_header)

        self.lbl_micro_label = QLabel("—")
        self.lbl_micro_label.setStyleSheet("font-size: 20px; font-weight: 800; color: #888888;")
        self.lbl_micro_label.setAlignment(Qt.AlignCenter)
        mi_layout.addWidget(self.lbl_micro_label)

        self.lbl_micro_status = QLabel("Stat: —") 
        self.lbl_micro_status.setStyleSheet("font-family: 'Consolas', monospace; font-size: 11px; color: #9e9e9e;")
        self.lbl_micro_status.setAlignment(Qt.AlignCenter)
        mi_layout.addWidget(self.lbl_micro_status)

        cards_layout.addWidget(micro_card)

        layout.addLayout(cards_layout)

        # --- Statistics ---
        self._add_separator(layout)
        
        stats_container = QWidget()
        stats_layout = QHBoxLayout(stats_container)
        stats_layout.setContentsMargins(0, 0, 0, 0)

        # Macro Col
        macro_col = QVBoxLayout()
        self._add_section_title(macro_col, "MACRO STATS")

        # Tweak colors slightly for dark theme visibility
        self.lbl_stats_positive = QLabel("Positive: —")
        self.lbl_stats_positive.setStyleSheet("color: #6a9955; font-family: 'Consolas', monospace; font-size: 11px;")
        macro_col.addWidget(self.lbl_stats_positive)

        self.lbl_stats_neutral = QLabel("Neutral: —")
        self.lbl_stats_neutral.setStyleSheet("color: #d7ba7d; font-family: 'Consolas', monospace; font-size: 11px;")
        macro_col.addWidget(self.lbl_stats_neutral)

        self.lbl_stats_negative = QLabel("Negative: —")
        self.lbl_stats_negative.setStyleSheet("color: #d16969; font-family: 'Consolas', monospace; font-size: 11px;")
        macro_col.addWidget(self.lbl_stats_negative)
        macro_col.addStretch()

        stats_layout.addLayout(macro_col)

        # Micro Col
        micro_col = QVBoxLayout()
        self._add_section_title(micro_col, "MICRO STATS")

        self.lbl_micro_event_count = QLabel("Events: —")
        self.lbl_micro_event_count.setStyleSheet("color: #cccccc; font-family: 'Consolas', monospace; font-size: 11px;")
        micro_col.addWidget(self.lbl_micro_event_count)

        self.lbl_micro_frame_count = QLabel("Frames: —")
        self.lbl_micro_frame_count.setStyleSheet("color: #cccccc; font-family: 'Consolas', monospace; font-size: 11px;")
        micro_col.addWidget(self.lbl_micro_frame_count)

        self.lbl_micro_pos = QLabel("Positive: —")
        self.lbl_micro_pos.setStyleSheet("color: #6a9955; font-family: 'Consolas', monospace; font-size: 11px;")
        micro_col.addWidget(self.lbl_micro_pos)

        self.lbl_micro_neg = QLabel("Negative: —")
        self.lbl_micro_neg.setStyleSheet("color: #d16969; font-family: 'Consolas', monospace; font-size: 11px;")
        micro_col.addWidget(self.lbl_micro_neg)
        micro_col.addStretch()

        stats_layout.addLayout(micro_col)
        
        layout.addWidget(stats_container)

        # --- ROI Vis ---
        self._add_separator(layout)
        self._add_section_title(layout, "ROI MOTION INTENSITY")
        
        self.roi_vis = RoiLineChartWidget()
        self.roi_vis.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.roi_vis, 1)

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

    # ==================
    # LOAD CSV
    # ==================

    def load_csv(self, file_path=None):
        if file_path is None or file_path is False:
            settings = QSettings("AplikasiSkripsi", "MicroExpression")
            last_dir = settings.value("last_result_dir", "results")
            
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Load Prediction Data",
                last_dir, "Prediction Package (*.result);;CSV Files (*.csv);;All Files (*)"
            )
            
            if file_path:
                settings.setValue("last_result_dir", os.path.dirname(file_path))
                
        if not file_path:
            return

        csv_path = file_path
        if file_path.lower().endswith('.result') or file_path.lower().endswith('.zip'):
            import zipfile
            import tempfile
            
            extract_dir = os.path.join(tempfile.gettempdir(), "expression_analyzer", os.path.splitext(os.path.basename(file_path))[0])
            os.makedirs(extract_dir, exist_ok=True)
            
            try:
                with zipfile.ZipFile(file_path, 'r') as zipf:
                    zipf.extractall(extract_dir)
                    
                csv_files = [f for f in os.listdir(extract_dir) if f.endswith('.csv')]
                if csv_files:
                    csv_path = os.path.join(extract_dir, csv_files[0])
                else:
                    self.lbl_csv_name.setText("Error: No CSV found in package")
                    return
            except Exception as e:
                self.lbl_csv_name.setText(f"Error loading package: {e}")
                return

        try:
            self.csv_data = load_csv_data(csv_path)
        except Exception as e:
            self.lbl_csv_name.setText(f"Error: {e}")
            return

        self.csv_path = csv_path
        self.total_frames = len(self.csv_data)

        csv_name = os.path.basename(csv_path)
        self.lbl_csv_name.setText(f"File: {csv_name}")
        self.lbl_total_frames.setText(f"Total Frames: {self.total_frames}")

        # Load metadata
        meta = load_meta(csv_path)
        source_path = meta.get("source", "—")
        is_folder = meta.get("is_folder", "False") == "True"
        fps = float(meta.get("fps", 30))

        if not os.path.isabs(source_path) and source_path != "—":
            source_path = os.path.join(os.path.dirname(csv_path), source_path)

        self.lbl_source_ref.setText(f"Source: {os.path.basename(source_path)}")

        self._compute_stats()

        self.playback_chart.set_data(self.csv_data)
        if hasattr(self, 'roi_vis'):
            self.roi_vis.set_data(self.csv_data)

        self.seek_slider.setEnabled(True)
        self.seek_slider.setMinimum(0)
        self.seek_slider.setMaximum(max(0, self.total_frames - 1))
        self.seek_slider.setValue(0)

        if self.total_frames > 0 and fps > 0:
            total_secs = self.total_frames / fps
            mins = int(total_secs // 60)
            secs = int(total_secs % 60)
            self.lbl_total_time.setText(f"{mins}:{secs:02d}")
        else:
            self.lbl_total_time.setText("0:00")

        if self.playback_worker:
            self.playback_worker.stop()

        self.playback_worker = PlaybackWorker()

        if os.path.exists(source_path):
            loaded = self.playback_worker.load_source(source_path, is_folder, fps)
            if loaded:
                self.playback_worker.frame_ready.connect(self._on_playback_frame)
                self.playback_worker.playback_finished.connect(self._on_playback_finished)
                self.playback_worker.start()

                self.btn_play.setEnabled(True)
                self.btn_stop.setEnabled(True)
                self.playback_worker.seek(0)
            else:
                self.btn_play.setEnabled(False)
                self.video_label.setText("Failed to open video source")
        else:
            self.btn_play.setEnabled(False)
            self.video_label.setText(
                f"Source file not found:\n{source_path}\n\n"
                "CSV data is still visible in the timeline."
            )

        if self.csv_data:
            self._update_current_prediction(0)

    def _compute_stats(self):
        if not self.csv_data:
            return

        total = len(self.csv_data)
        counts = {"positive": 0, "neutral": 0, "negative": 0}
        micro_frame_counts = {"positive": 0, "negative": 0}

        for row in self.csv_data:
            label = row["macro_label"].lower()
            if label in counts:
                counts[label] += 1

            micro = row.get("micro_label", "").strip().lower()
            if micro in micro_frame_counts:
                micro_frame_counts[micro] += 1

        # Macro stats
        self.lbl_stats_positive.setText(
            f"Pos: {counts['positive']} ({counts['positive']/total*100:.1f}%)"
        )
        self.lbl_stats_neutral.setText(
            f"Neu: {counts['neutral']} ({counts['neutral']/total*100:.1f}%)"
        )
        self.lbl_stats_negative.setText(
            f"Neg: {counts['negative']} ({counts['negative']/total*100:.1f}%)"
        )

        # Micro stats
        micro_events = 0
        in_event = False
        for row in self.csv_data:
            micro = row.get("micro_label", "").strip().lower()
            has_micro = micro in ("positive", "negative")
            if has_micro and not in_event:
                micro_events += 1
                in_event = True
            elif not has_micro:
                in_event = False

        micro_total_frames = micro_frame_counts['positive'] + micro_frame_counts['negative']

        self.lbl_micro_event_count.setText(f"Events: {micro_events}")
        self.lbl_micro_frame_count.setText(
            f"Frames: {micro_total_frames} ({micro_total_frames/total*100:.1f}%)"
        )
        self.lbl_micro_pos.setText(
            f"Pos: {micro_frame_counts['positive']} f"
        )
        self.lbl_micro_neg.setText(
            f"Neg: {micro_frame_counts['negative']} f"
        )

    # ==================
    # PLAYBACK CONTROLS
    # ==================

    def toggle_play(self):
        if not self.playback_worker:
            return

        if self.is_playing:
            self.playback_worker.pause()
            self.btn_play.setText("▶")
            self.is_playing = False
        else:
            self.playback_worker.play()
            self.btn_play.setText("⏸")
            self.is_playing = True

    def stop_playback(self):
        if self.playback_worker:
            self.playback_worker.pause()
            self.playback_worker.seek(0)

        self.btn_play.setText("▶")
        self.is_playing = False
        self.seek_slider.setValue(0)

    def _on_speed_changed(self):
        speed = self.speed_combo.currentData()
        if self.playback_worker and speed:
            self.playback_worker.set_speed(speed)

    def _on_slider_seek(self, value):
        if self.playback_worker:
            self.playback_worker.seek(value)
        self._update_current_prediction(value)
        self.playback_chart.set_cursor(value)

    def _on_chart_seek(self, frame_idx):
        if self.playback_worker:
            self.playback_worker.seek(frame_idx)
        self.seek_slider.blockSignals(True)
        self.seek_slider.setValue(frame_idx)
        self.seek_slider.blockSignals(False)
        self._update_current_prediction(frame_idx)

    def _on_playback_frame(self, frame_rgb, frame_idx):
        h, w, ch = frame_rgb.shape
        bpl = ch * w
        image = QImage(frame_rgb.data, w, h, bpl, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)

        scaled = pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.FastTransformation
        )
        self.video_label.setPixmap(scaled)

        self.seek_slider.blockSignals(True)
        self.seek_slider.setValue(frame_idx)
        self.seek_slider.blockSignals(False)

        self._update_current_prediction(frame_idx)
        self.playback_chart.set_cursor(frame_idx)
        self.update_scroll_to_cursor()

        if self.csv_data and 0 <= frame_idx < len(self.csv_data):
            ts = self.csv_data[frame_idx]["timestamp_ms"]
            total_secs = ts / 1000
            mins = int(total_secs // 60)
            secs = int(total_secs % 60)
            self.lbl_time.setText(f"{mins}:{secs:02d}")

    def _on_playback_finished(self):
        self.btn_play.setText("▶")
        self.is_playing = False

    def _update_current_prediction(self, frame_idx):
        if not self.csv_data or frame_idx >= len(self.csv_data):
            return

        row = self.csv_data[frame_idx]
        label = row["macro_label"]
        label_lower = label.lower()
        conf = row["macro_confidence"]

        self.lbl_frame_num.setText(f"Frame: {frame_idx}")

        # Macro color - softer modern palette
        if label_lower == "positive":
            color = "#6a9955"
        elif label_lower == "negative":
            color = "#d16969"
        elif label_lower == "neutral":
            color = "#d7ba7d"
        else:
            color = "#888888"

        self.lbl_macro_label.setText(label.upper())
        self.lbl_macro_label.setStyleSheet(f"""
            font-family: 'Segoe UI', 'Roboto', sans-serif;
            font-size: 20px; font-weight: 800; color: {color};
        """)
        self.lbl_macro_conf.setText(f"Conf: {conf:.4f}")

        # Micro label
        micro = row.get('micro_label', '').strip()
        micro_conf = row.get('micro_confidence', 0)
        if micro and micro.lower() not in ('', 'n/a'):
            micro_lower = micro.lower()
            if micro_lower == 'positive':
                micro_color = '#6a9955'
            elif micro_lower == 'negative':
                micro_color = '#d16969'
            else:
                micro_color = '#aaaaaa'
            self.lbl_micro_label.setText(micro.upper())
            self.lbl_micro_label.setStyleSheet(f"""
                font-family: 'Segoe UI', 'Roboto', sans-serif;
                font-size: 20px; font-weight: 800; color: {micro_color};
            """)
            self.lbl_micro_status.setText(f"Conf: {micro_conf:.4f}")
            self.lbl_micro_status.setStyleSheet("font-family: 'Consolas', monospace; font-size: 11px; color: #9e9e9e;")
        else:
            self.lbl_micro_label.setText("—")
            self.lbl_micro_label.setStyleSheet("font-family: 'Segoe UI', sans-serif; font-size: 20px; font-weight: 800; color: #444444;")
            self.lbl_micro_status.setText("Not detected")
            self.lbl_micro_status.setStyleSheet("font-family: 'Consolas', monospace; font-size: 11px; color: #666666;")
            
        # Update ROI Visualization Cursor
        if hasattr(self, 'roi_vis'):
            self.roi_vis.set_cursor(frame_idx)
    
    def update_scroll_to_cursor(self):
        # 1. Pastikan data ada
        if self.playback_chart.total_frames <= 0:
            return

        if self.playback_chart.is_dragging:
            return
        
        # 2. Ambil referensi scroll bar horizontal
        scroll_bar = self.scroll_area.horizontalScrollBar()
        
        # 3. Hitung posisi X kursor dalam pixel
        chart_width = self.playback_chart.width()
        cursor_x = (self.playback_chart.cursor_pos / self.playback_chart.total_frames) * chart_width

        # 4. Hitung posisi scroll agar kursor berada di tengah
        viewport_width = self.scroll_area.viewport().width()
        target_scroll = cursor_x - (viewport_width / 2)

        # 5. Terapkan ke scroll bar
        scroll_bar.setValue(int(target_scroll))

    # ==================
    # CLEANUP
    # ==================

    def cleanup(self):
        if self.playback_worker:
            self.playback_worker.stop()
            self.playback_worker = None