import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QFileDialog, QSlider, QComboBox, QSizePolicy
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor
import cv2

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
            color: #555;
            font-size: 14px;
            font-weight: bold;
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
        chart_header.setStyleSheet("background-color: #1e1e1e; border-bottom: 1px solid #3a3a3a;")
        ch_layout = QHBoxLayout(chart_header)
        ch_layout.setContentsMargins(12, 0, 12, 0)
        ch_title = QLabel("EMOTION TIMELINE")
        ch_title.setStyleSheet("color: #777; font-size: 10px; font-weight: bold; letter-spacing: 2px;")
        ch_layout.addWidget(ch_title)
        chart_layout.addWidget(chart_header)

        self.playback_chart = PlaybackChart()
        self.playback_chart.seek_requested.connect(self._on_chart_seek)
        chart_layout.addWidget(self.playback_chart)

        left_layout.addWidget(chart_container, 1)

        main_layout.addWidget(left_panel, 4)

        # ====== RIGHT SIDEBAR ======
        sidebar = self._create_sidebar()
        main_layout.addWidget(sidebar, 1)

    def _create_controls_bar(self):
        bar = QFrame()
        bar.setFixedHeight(44)
        bar.setStyleSheet("""
            QFrame { background-color: #1e1e1e; border-top: 1px solid #3a3a3a; }
            QLabel { color: #999; font-size: 10px; }
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
            QPushButton:disabled { color: #444; }
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
            QPushButton:disabled { color: #444; }
        """)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_playback)
        layout.addWidget(self.btn_stop)

        # Current time
        self.lbl_time = QLabel("0:00")
        self.lbl_time.setStyleSheet("color: #ccc; font-family: Consolas; font-size: 11px; min-width: 36px;")
        layout.addWidget(self.lbl_time)

        # Seek slider — Adobe style
        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setEnabled(False)
        self.seek_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: none;
                height: 4px;
                background: #383838;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #2d8ceb;
                width: 12px;
                height: 12px;
                margin: -4px 0;
                border-radius: 6px;
            }
            QSlider::handle:horizontal:hover {
                background: #4aa3f5;
            }
            QSlider::sub-page:horizontal {
                background: #2d8ceb;
                border-radius: 2px;
            }
        """)
        self.seek_slider.valueChanged.connect(self._on_slider_seek)
        layout.addWidget(self.seek_slider, 1)

        # Total time
        self.lbl_total_time = QLabel("0:00")
        self.lbl_total_time.setStyleSheet("color: #777; font-family: Consolas; font-size: 11px; min-width: 36px;")
        layout.addWidget(self.lbl_total_time)

        # Speed selector
        speed_label = QLabel("Speed:")
        layout.addWidget(speed_label)

        self.speed_combo = QComboBox()
        self.speed_combo.setStyleSheet("""
            QComboBox {
                background-color: #383838; color: #ccc;
                border: 1px solid #4a4a4a; border-radius: 3px;
                padding: 3px 6px; font-size: 10px;
            }
            QComboBox QAbstractItemView {
                background-color: #383838; color: #ccc;
                selection-background-color: #2d8ceb;
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
        sidebar.setMaximumWidth(340)
        sidebar.setStyleSheet("""
            QFrame { background-color: #2d2d2d; border: none; border-left: 1px solid #3a3a3a; }
            QLabel { color: #ccc; font-size: 11px; }
        """)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(5)

        # Title
        title = QLabel("ANALYSIS")
        title.setStyleSheet("""
            font-size: 11px; font-weight: bold; color: #2d8ceb;
            letter-spacing: 2px;
        """)
        layout.addWidget(title)

        # Load CSV button
        self.btn_load_csv = QPushButton("  Load CSV File")
        self.btn_load_csv.setStyleSheet("""
            QPushButton {
                background-color: #2d8ceb; color: white;
                border: none; border-radius: 3px;
                padding: 10px; font-size: 12px; font-weight: bold;
            }
            QPushButton:hover { background-color: #4aa3f5; }
        """)
        self.btn_load_csv.clicked.connect(self.load_csv)
        layout.addWidget(self.btn_load_csv)

        # --- Info section ---
        self._add_separator(layout)
        self._add_section_title(layout, "DATA INFO")

        self.lbl_csv_name = QLabel("CSV: —")
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
        self.lbl_frame_num.setStyleSheet("font-family: Consolas; font-size: 11px;")
        layout.addWidget(self.lbl_frame_num)

       # Macro card
        macro_card = QFrame()
        macro_card.setObjectName("macroCard")
        macro_card.setStyleSheet("""
            QFrame#macroCard {          
                background-color: #232323;
                border: 1px solid #3a3a3a;
                border-radius: 3px;
            }
        """)
        mc_layout = QVBoxLayout(macro_card)
        mc_layout.setContentsMargins(10, 6, 10, 6)
        mc_layout.setSpacing(3)

        mc_header = QLabel("MACRO EXPRESSION")
        mc_header.setStyleSheet("color: #777; font-size: 9px; font-weight: bold; letter-spacing: 1px;")
        mc_layout.addWidget(mc_header)

        self.lbl_macro_label = QLabel("—")
        self.lbl_macro_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #888;")
        self.lbl_macro_label.setAlignment(Qt.AlignCenter)
        mc_layout.addWidget(self.lbl_macro_label)

        self.lbl_macro_conf = QLabel("Confidence: —")
        self.lbl_macro_conf.setStyleSheet("font-family: Consolas; font-size: 10px; color: #888;")
        self.lbl_macro_conf.setAlignment(Qt.AlignCenter)
        mc_layout.addWidget(self.lbl_macro_conf)

        layout.addWidget(macro_card)

        # Micro card
        micro_card = QFrame()
        micro_card.setObjectName("microCard")
        micro_card.setStyleSheet("""
            QFrame#microCard {
                background-color: #232323;
                border: 1px solid #3a3a3a;
                border-radius: 3px;
            }
        """)
        mi_layout = QVBoxLayout(micro_card)
        mi_layout.setContentsMargins(10, 6, 10, 6)
        mi_layout.setSpacing(3)

        mi_header = QLabel("MICRO EXPRESSION")
        mi_header.setStyleSheet("color: #777; font-size: 9px; font-weight: bold; letter-spacing: 1px;")
        mi_layout.addWidget(mi_header)

        self.lbl_micro_label = QLabel("—")
        self.lbl_micro_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #888;")
        self.lbl_micro_label.setAlignment(Qt.AlignCenter)
        mi_layout.addWidget(self.lbl_micro_label)

        self.lbl_micro_status = QLabel("Status: —")
        self.lbl_micro_status.setStyleSheet("font-family: Consolas; font-size: 10px; color: #888;")
        self.lbl_micro_status.setAlignment(Qt.AlignCenter)
        mi_layout.addWidget(self.lbl_micro_status)

        layout.addWidget(micro_card)

        # --- Stats: Macro ---
        self._add_separator(layout)
        self._add_section_title(layout, "MACRO STATISTICS")

        self.lbl_stats_positive = QLabel("Positive: —")
        self.lbl_stats_positive.setStyleSheet("color: #4caf50;")
        layout.addWidget(self.lbl_stats_positive)

        self.lbl_stats_neutral = QLabel("Neutral: —")
        self.lbl_stats_neutral.setStyleSheet("color: #e6a817;")
        layout.addWidget(self.lbl_stats_neutral)

        self.lbl_stats_negative = QLabel("Negative: —")
        self.lbl_stats_negative.setStyleSheet("color: #e74c3c;")
        layout.addWidget(self.lbl_stats_negative)

        # --- Stats: Micro ---
        self._add_separator(layout)
        self._add_section_title(layout, "MICRO STATISTICS")

        self.lbl_micro_event_count = QLabel("Events Detected: —")
        self.lbl_micro_event_count.setStyleSheet("color: #aaa;")
        layout.addWidget(self.lbl_micro_event_count)

        self.lbl_micro_frame_count = QLabel("Frames Affected: —")
        self.lbl_micro_frame_count.setStyleSheet("color: #aaa;")
        layout.addWidget(self.lbl_micro_frame_count)

        self.lbl_micro_pos = QLabel("Positive: —")
        self.lbl_micro_pos.setStyleSheet("color: #4caf50;")
        layout.addWidget(self.lbl_micro_pos)

        self.lbl_micro_neg = QLabel("Negative: —")
        self.lbl_micro_neg.setStyleSheet("color: #e74c3c;")
        layout.addWidget(self.lbl_micro_neg)

        layout.addStretch()
        return sidebar

    def _add_separator(self, layout):
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("background-color: #3a3a3a; max-height: 1px; margin-top: 3px; margin-bottom: 1px;")
        layout.addWidget(sep)

    def _add_section_title(self, layout, text):
        lbl = QLabel(text)
        lbl.setStyleSheet("""
            font-size: 10px; font-weight: bold; color: #999;
            letter-spacing: 2px; margin-top: 2px;
        """)
        layout.addWidget(lbl)

    # ==================
    # LOAD CSV
    # ==================

    def load_csv(self, csv_path=None):
        if csv_path is None or csv_path is False:
            csv_path, _ = QFileDialog.getOpenFileName(
                self, "Load Prediction CSV",
                "results", "CSV Files (*.csv);;All Files (*)"
            )
        if not csv_path:
            return

        try:
            self.csv_data = load_csv_data(csv_path)
        except Exception as e:
            self.lbl_csv_name.setText(f"Error: {e}")
            return

        self.csv_path = csv_path
        self.total_frames = len(self.csv_data)

        csv_name = os.path.basename(csv_path)
        self.lbl_csv_name.setText(f"CSV: {csv_name}")
        self.lbl_total_frames.setText(f"Total Frames: {self.total_frames}")

        # Load metadata
        meta = load_meta(csv_path)
        source_path = meta.get("source", "—")
        is_folder = meta.get("is_folder", "False") == "True"
        fps = float(meta.get("fps", 30))

        self.lbl_source_ref.setText(f"Source: {os.path.basename(source_path)}")

        self._compute_stats()

        self.playback_chart.set_data(self.csv_data)

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
            f"Positive: {counts['positive']}  ({counts['positive']/total*100:.1f}%)"
        )
        self.lbl_stats_neutral.setText(
            f"Neutral: {counts['neutral']}  ({counts['neutral']/total*100:.1f}%)"
        )
        self.lbl_stats_negative.setText(
            f"Negative: {counts['negative']}  ({counts['negative']/total*100:.1f}%)"
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

        self.lbl_micro_event_count.setText(f"Events Detected: {micro_events}")
        self.lbl_micro_frame_count.setText(
            f"Frames Affected: {micro_total_frames}  ({micro_total_frames/total*100:.1f}%)"
        )
        self.lbl_micro_pos.setText(
            f"Positive: {micro_frame_counts['positive']} frames"
        )
        self.lbl_micro_neg.setText(
            f"Negative: {micro_frame_counts['negative']} frames"
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
        if 0 <= frame_idx < len(self.csv_data):
            row = self.csv_data[frame_idx]
            x1, y1 = row["face_x1"], row["face_y1"]
            x2, y2 = row["face_x2"], row["face_y2"]

            if x1 > 0 or y1 > 0 or x2 > 0 or y2 > 0:
                label = row["macro_label"]
                label_lower = label.lower()
                if label_lower == "positive":
                    color = (76, 175, 80)    # green RGB
                elif label_lower == "negative":
                    color = (231, 76, 60)    # red RGB
                elif label_lower == "neutral":
                    color = (230, 168, 23)   # amber RGB
                else:
                    color = (136, 136, 136)

                cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), color, 2)

                display_text = f"{label} ({row['macro_confidence']:.2f})"
                micro = row.get('micro_label', '').strip()
                if micro and micro.lower() not in ('', 'n/a'):
                    display_text += f" | Micro: {micro}"

                cv2.putText(
                    frame_rgb, display_text,
                    (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )

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

        # Macro color
        if label_lower == "positive":
            color = "#4caf50"
        elif label_lower == "negative":
            color = "#e74c3c"
        elif label_lower == "neutral":
            color = "#e6a817"
        else:
            color = "#888888"

        self.lbl_macro_label.setText(label.upper())
        self.lbl_macro_label.setStyleSheet(f"""
            font-size: 18px; font-weight: bold; color: {color};
        """)
        self.lbl_macro_conf.setText(f"Confidence: {conf:.4f}")

        # Micro label
        micro = row.get('micro_label', '').strip()
        micro_conf = row.get('micro_confidence', 0)
        if micro and micro.lower() not in ('', 'n/a'):
            micro_lower = micro.lower()
            if micro_lower == 'positive':
                micro_color = '#4caf50'
            elif micro_lower == 'negative':
                micro_color = '#e74c3c'
            else:
                micro_color = '#aaa'
            self.lbl_micro_label.setText(micro.upper())
            self.lbl_micro_label.setStyleSheet(f"""
                font-size: 16px; font-weight: bold; color: {micro_color};
            """)
            self.lbl_micro_status.setText(f"Confidence: {micro_conf:.4f}")
            self.lbl_micro_status.setStyleSheet(f"font-family: Consolas; font-size: 10px; color: {micro_color};")
        else:
            self.lbl_micro_label.setText("—")
            self.lbl_micro_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #444;")
            self.lbl_micro_status.setText("Not detected")
            self.lbl_micro_status.setStyleSheet("font-size: 10px; color: #555;")

    # ==================
    # CLEANUP
    # ==================

    def cleanup(self):
        if self.playback_worker:
            self.playback_worker.stop()
            self.playback_worker = None
