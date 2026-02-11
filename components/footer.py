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
        
        lbl_value = QLabel("NEUTRAL (0.0%)") # Default text
        lbl_value.setObjectName("valueLabel")
        lbl_value.setAlignment(Qt.AlignRight)

        header_row.addWidget(lbl_title)
        header_row.addStretch()
        header_row.addWidget(lbl_value)

        # Pastikan LiveLineChart Anda support range -100 sampai 100
        chart = LiveLineChart(max_points=100) 

        layout.addLayout(header_row)
        layout.addWidget(chart)

        return {
            "container": container,
            "chart": chart,
            "value_label": lbl_value,
            "title_label": lbl_title
        }

    # =========================================
    # LOGIKA BARU UNTUK MENERIMA DATA REAL
    # =========================================
    def update_prediction(self, label, confidence):
        """
        Menerima Label (Negative/Neutral/Positive) dan Confidence (0.0 - 1.0)
        Mengubahnya menjadi nilai visual untuk grafik.
        """
        
        # 1. Konversi Label & Confidence menjadi Score (-100 s/d 100)
        # Confidence 0.8 Positive -> +80
        # Confidence 0.8 Negative -> -80
        # Neutral -> 0
        
        score = 0.0
        display_color = "#ffaa33" # Default Orange (Neutral)

        if label == "Positive":
            score = confidence * 100
            display_color = "#00d27a" # Green
        elif label == "Negative":
            score = -confidence * 100 # Menjadi negatif untuk grafik turun
            display_color = "#ff4c4c" # Red
        else: # Neutral
            score = 0
            display_color = "#ffaa33" # Orange

        # 2. Update UI Bagian Macro (Asumsi model Anda adalah Macro)
        self._update_ui_section(self.macro_section, label, score, display_color)
        
        # (Opsional) Jika belum ada Micro model, bisa di set 0 atau ikut Macro
        self._update_ui_section(self.micro_section, "N/A", 0, "#888")

    def _update_ui_section(self, section, label_text, score_val, color_hex):
        # Update Grafik (Kirim angka -100 s/d 100)
        section["chart"].update_value(score_val)

        # Update Teks: Contoh "POSITIVE (85.2%)"
        # Kita ambil nilai absolute untuk persentase agar tidak muncul -85%
        abs_score = abs(score_val)
        section["value_label"].setText(f"{label_text.upper()} ({abs_score:.1f}%)")
        
        # Update Warna
        section["value_label"].setStyleSheet(f"color: {color_hex};")
        section["title_label"].setStyleSheet(f"color: {color_hex}; font-weight: bold;")