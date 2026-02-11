from collections import deque
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import (
    QPainter, QColor, QPen, QBrush, 
    QPolygonF
)

class LiveLineChart(QWidget):
    def __init__(self, max_points=100):
        super().__init__()
        self.setMinimumHeight(60)
        
        # Buffer data
        self.max_points = max_points
        self.data = deque([0] * max_points, maxlen=max_points)

    def update_value(self, value):
        val = max(-100, min(100, value))
        self.data.append(val)
        self.update() # Trigger repaint

    def get_color(self, value):
        """Menentukan warna berdasarkan nilai"""
        if value < -30:
            return QColor("#ff4c4c") # Merah
        elif value > 30:
            return QColor("#00d27a") # Hijau
        else:
            return QColor("#ffaa33") # Orange/Kuning

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()

        # -----------------------------
        # 1. Background Area (Gelap)
        # -----------------------------
        painter.fillRect(0, 0, w, h, QColor("#1e1e1e"))

        # -----------------------------
        # 2. Grid Lines (Garis Bantu)
        # -----------------------------
        pen_grid = QPen(QColor("#333333"), 1)
        pen_grid.setStyle(Qt.DashLine)
        painter.setPen(pen_grid)
        
        y_zero = h / 2
        y_top = h * 0.25 
        y_bot = h * 0.75 
        
        painter.drawLine(0, int(y_zero), w, int(y_zero))
        painter.drawLine(0, int(y_top), w, int(y_top))
        painter.drawLine(0, int(y_bot), w, int(y_bot))

        # Helper: Map nilai (-100 s/d 100) ke Y coordinate
        def get_y(val):
            normalized = (val + 100) / 200 
            return h - (normalized * h)

        # Hitung jarak X antar titik
        step_x = w / (self.max_points - 1) if self.max_points > 1 else 0
        
        # Ubah deque ke list untuk akses index
        data_list = list(self.data)

        # -----------------------------
        # 3. Gambar Segmen (Fill & Line)
        # -----------------------------
        # Kita loop setiap pasangan titik
        for i in range(len(data_list) - 1):
            val1 = data_list[i]
            val2 = data_list[i+1]
            
            x1 = i * step_x
            y1 = get_y(val1)
            
            x2 = (i+1) * step_x
            y2 = get_y(val2)
            
            # Tentukan warna berdasarkan titik tujuan (val2)
            base_color = self.get_color(val2)
            
            # --- A. Gambar FILL (Background bawah garis) ---
            # Kita buat Polygon (Trapesium):
            # (x1, y1) -> (x2, y2) -> (x2, h) -> (x1, h)
            polygon = QPolygonF()
            polygon.append(QPointF(x1, y1)) # Kiri Atas
            polygon.append(QPointF(x2, y2)) # Kanan Atas
            polygon.append(QPointF(x2, h))  # Kanan Bawah (Dasar)
            polygon.append(QPointF(x1, h))  # Kiri Bawah (Dasar)
            
            # Set warna transparan untuk fill
            fill_color = QColor(base_color)
            fill_color.setAlpha(50) # Transparansi (0-255)
            
            painter.setPen(Qt.NoPen)
            painter.setBrush(fill_color)
            painter.drawPolygon(polygon)

            # --- B. Gambar GARIS (Line Stroke) ---
            # Gambar garis solid di atas fill
            pen = QPen(base_color, 2)
            pen.setCapStyle(Qt.RoundCap) # Ujung bulat agar sambungan halus
            
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))
            
        # -----------------------------
        # 4. Indikator Titik Terakhir (Bulatan)
        # -----------------------------
        if data_list:
            last_val = data_list[-1]
            last_x = w - 3
            last_y = get_y(last_val)
            
            painter.setPen(Qt.NoPen)
            painter.setBrush(self.get_color(last_val))
            painter.drawEllipse(QPointF(last_x, last_y), 4, 4)