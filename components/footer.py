from PySide6.QtWidgets import QFrame, QVBoxLayout, QLabel, QHBoxLayout
from PySide6.QtCore import Qt
from components.line_chart import LiveLineChart 
# Hapus import random dan QTimer karena tidak dipakai lagi

class PredictionFooter(QFrame):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(285) 

        # --- STYLING (Sama seperti kode Anda) ---
        self.setStyleSheet("""
            QFrame { background-color: #252526; border: none; }
            QLabel { color: #dddddd; font-family: 'Segoe UI', sans-serif; }
            #title { font-size: 14px; font-weight: bold; color: #888; letter-spacing: 1px; }
            #sectionTitle { font-size: 12px; font-weight: bold; }
            #valueLabel { font-size: 14px; font-weight: bold; font-family: 'Consolas', 'Monospace'; }
        """)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 20)
        main_layout.setSpacing(1)

        # ===== HEADER BAR =====
        header = QFrame()
        header.setFixedHeight(30)
        header.setStyleSheet("background-color: #1e1e1e; border-bottom: 1px solid #333;")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(15, 0, 15, 0)
        title = QLabel("REAL-TIME EMOTION ANALYSIS")
        title.setObjectName("title")
        header_layout.addWidget(title)
        main_layout.addWidget(header)

        # ===== MACRO SECTION =====
        self.macro_section = self.create_chart_section("MACRO EXPRESSION")
        main_layout.addWidget(self.macro_section["container"])

        # ===== MICRO SECTION =====
        # (Opsional: Jika belum ada model Micro, biarkan kosong atau duplikasi)
        self.micro_section = self.create_chart_section("MICRO EXPRESSION")
        main_layout.addWidget(self.micro_section["container"])

    def create_chart_section(self, title_text):
        """Helper untuk membuat UI Section"""
        container = QFrame()
        container.setStyleSheet("border-bottom: 1px solid #333;")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(15, 8, 15, 8)
        layout.setSpacing(5)

        header_row = QHBoxLayout()
        lbl_title = QLabel(title_text)
        lbl_title.setObjectName("sectionTitle")
        
        lbl_value = QLabel("NEUTRAL") # Default text
        lbl_value.setObjectName("valueLabel")
        lbl_value.setAlignment(Qt.AlignRight)

        header_row.addWidget(lbl_title)
        header_row.addStretch()
        header_row.addWidget(lbl_value)

        # Pastikan LiveLineChart Anda support range -100 sampai 100
        chart = LiveLineChart(max_points=1000) 

        layout.addLayout(header_row)
        layout.addWidget(chart)

        return {
            "container": container,
            "chart": chart,
            "value_label": lbl_value,
            "title_label": lbl_title
        }

    # =========================================
    # LOGIKA UNTUK MENERIMA DATA REAL
    # =========================================
    def update_prediction(self, label, confidence):
        score = 0.0
        display_color = "#888888" # Default Gray (Not Detected / Unknown)
        display_text = "NOT DETECTED"

        # Logika Penentuan Warna & Nilai
        if label == "Positive":
            score = confidence * 100
            display_color = "#00d27a" # Green
            display_text = f"POSITIVE ({abs(score):.1f}%)"
            
        elif label == "Negative":
            score = -confidence * 100
            display_color = "#ff4c4c" # Red
            display_text = f"NEGATIVE ({abs(score):.1f}%)"
            
        elif label == "Neutral":
            score = 0 # Tetap 0, tapi warnanya beda dengan Not Detected
            display_color = "#ffaa33" # Orange
            display_text = "NEUTRAL"
            
        else:
            # Case: "Not Detected" atau label sampah lainnya
            score = 0
            display_color = "#888888" # Abu-abu
            display_text = label.upper()

        # Update Macro Section
        self._update_ui_section(self.macro_section, display_text, score, display_color)
        
        # Micro Section (Dummy/Default)
        self._update_ui_section(self.micro_section, "N/A", 0, "#444444")

    def _update_ui_section(self, section, text_val, score_val, color_hex):
        # Update text & warna label
        section["value_label"].setText(text_val)
        section["value_label"].setStyleSheet(f"color: {color_hex};")
        # section["title_label"].setStyleSheet(f"color: {color_hex}; font-weight: bold;")
        
        # PENTING: Kirim score DAN warna ke chart
        section["chart"].update_value(score_val, color_hex)