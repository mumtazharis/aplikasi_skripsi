from collections import deque
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import (
    QPainter, QColor, QPen, 
    QPolygonF
)

class LiveLineChart(QWidget):
    def __init__(self, max_points=100):
        super().__init__()
        self.setMinimumHeight(60)
        
        self.max_points = max_points
        
        # Default color: Gray
        default_color = QColor("#888888")
        
        # Buffer sekarang menyimpan TUPLE: (value, QColor)
        # Inisialisasi dengan 0 dan warna abu-abu
        self.data = deque([(0, default_color)] * max_points, maxlen=max_points)

    def update_value(self, value, color_hex):
        """
        Menerima value (-100 s/d 100) DAN kode warna hex.
        """
        val = max(-100, min(100, value))
        col = QColor(color_hex)
        
        # Simpan pasangan (nilai, warna)
        self.data.append((val, col))
        self.update() 

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()

        # 1. Background
        painter.fillRect(0, 0, w, h, QColor("#1e1e1e"))

        # 2. Grid Lines
        pen_grid = QPen(QColor("#333333"), 1)
        pen_grid.setStyle(Qt.DashLine)
        painter.setPen(pen_grid)
        
        y_zero = h / 2
        y_top = h * 0.25 
        y_bot = h * 0.75 
        
        painter.drawLine(0, int(y_zero), w, int(y_zero))
        painter.drawLine(0, int(y_top), w, int(y_top))
        painter.drawLine(0, int(y_bot), w, int(y_bot))

        # Helper Y coordinate
        def get_y(val):
            normalized = (val + 100) / 200 
            return h - (normalized * h)

        # Jarak X antar titik
        step_x = w / (self.max_points - 1) if self.max_points > 1 else 0
        
        # Convert deque ke list agar bisa diakses index-nya
        data_list = list(self.data)

        # 3. Gambar Segmen (Looping)
        for i in range(len(data_list) - 1):
            # Ambil nilai dan warna dari tuple
            val1, _ = data_list[i]      # Warna titik awal tidak terlalu penting untuk garis ke kanan
            val2, col2 = data_list[i+1] # Kita gunakan warna titik tujuan untuk segmen ini
            
            x1 = i * step_x
            y1 = get_y(val1)
            
            x2 = (i+1) * step_x
            y2 = get_y(val2)
            
            # --- A. FILL (Area) ---
            polygon = QPolygonF()
            polygon.append(QPointF(x1, y1))
            polygon.append(QPointF(x2, y2))
            polygon.append(QPointF(x2, h))
            polygon.append(QPointF(x1, h))
            
            fill_color = QColor(col2)
            fill_color.setAlpha(40) # Transparansi fill
            
            painter.setPen(Qt.NoPen)
            painter.setBrush(fill_color)
            painter.drawPolygon(polygon)

            # --- B. LINE (Stroke) ---
            pen = QPen(col2, 2) # Gunakan warna yang tersimpan
            pen.setCapStyle(Qt.RoundCap)
            
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))
            
        # 4. Indikator Titik Terakhir (Bulatan)
        if data_list:
            last_val, last_col = data_list[-1]
            last_x = w - 3
            last_y = get_y(last_val)
            
            painter.setPen(Qt.NoPen)
            painter.setBrush(last_col) # Gunakan warna terakhir
            painter.drawEllipse(QPointF(last_x, last_y), 4, 4)