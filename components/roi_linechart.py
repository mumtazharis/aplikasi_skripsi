from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QPainter, QColor, QPen, QPolygonF
from PySide6.QtCore import Qt, QRectF, QPointF

class RoiLineChartWidget(QWidget):
    """
    Widget untuk menggambar Grafik Energy ROI (Global dan Region).
    Menggunakan konsep Sliding Window layaknya Oscilloscope.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(220, 300)
        
        # Jendela frame yang divisualisasikan dalam satu waktu
        self.window_size = 120
        
        # Daftar ROI sama dengan yang diekstrak di CSV
        self.roi_keys = [
            "area_dahi", "area_alis_kanan", "area_alis_kiri", 
            "area_antara_alis", 
            "area_pipi_kanan", "area_pipi_kiri", "area_hidung", 
            "area_mulut_kanan", "area_mulut_kiri"
        ]
        
        # Warna-warni kontras tinggi untuk membedakan hingga 11+ ROI
        self.roi_colors = [
            "#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231", 
            "#911eb4", "#42d4f4", "#f032e6", "#bfef45"
        ]
        
        self.history = {k: [] for k in self.roi_keys}
        self.global_history = []
        self.total_frames = 0
        self.current_frame = 0
        
        self.global_max_y = 0.001
        self.region_max_y = 0.001
        
        self.visible_lines = {k: True for k in self.roi_keys}
        self.legend_rects = []
        
        self.setCursor(Qt.PointingHandCursor)

    def set_data(self, full_csv_data):
        """
        Membaca seluruh file CSV di memori untuk dirender sebagai garis panjang.
        """
        self.history = {k: [] for k in self.roi_keys}
        self.global_history = []
        self.total_frames = len(full_csv_data)
        self.current_frame = 0
        
        for row in full_csv_data:
            global_sum = 0.0
            for k in self.roi_keys:
                try:
                    val = float(row.get(k, 0.0))
                except ValueError:
                    val = 0.0
                self.history[k].append(val)
                global_sum += val
            self.global_history.append(global_sum)
            
        # Hitung max skala statis untuk keseluruhan data
        self.global_max_y = max(self.global_history) * 1.2 if self.global_history else 0.001
        self.global_max_y = max(self.global_max_y, 0.001)
        
        max_reg = 0.001
        for k in self.roi_keys:
            if self.history[k]:
                max_reg = max(max_reg, max(self.history[k]))
        self.region_max_y = max_reg * 1.2
            
        self.update()

    def set_cursor(self, frame_idx):
        """
        Menggeser cursor window.
        """
        self.current_frame = max(0, min(frame_idx, self.total_frames - 1))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        
        w = self.width()
        h = self.height()
        
        # Latar Belakang
        painter.fillRect(0, 0, w, h, QColor("#232323"))
        
        if self.total_frames == 0:
            painter.setPen(QColor("#777777"))
            painter.drawText(self.rect(), Qt.AlignCenter, "No ROI Data")
            return
            
        # Potong area render jadi dua atas-bawah + ruang legend
        legend_h = 85
        available_h = h - legend_h
        global_h = available_h // 2
        region_h = available_h - global_h
        
        # --- 1. GLOBAL CHART ---
        self._draw_chart(painter, 0, 0, w, global_h, "GLOBAL MOTION", [self.global_history], ["#2d8ceb"], self.global_max_y, [True])
        
        # --- 2. REGION CHART ---
        region_history_list = [self.history[k] for k in self.roi_keys]
        region_visible_mask = [self.visible_lines[k] for k in self.roi_keys]
        self._draw_chart(painter, 0, global_h, w, region_h, "REGION MOTION", region_history_list, self.roi_colors, self.region_max_y, region_visible_mask)
        
        # --- 3. LEGEND ---
        self._draw_legend(painter, 0, h - legend_h, w, legend_h)
        
    def _draw_legend(self, painter, x_offset, y_offset, w, h_legend):
        self.legend_rects.clear()
        painter.setPen(QColor("#cccccc"))
        font = painter.font()
        font.setPointSize(7)
        painter.setFont(font)
        
        col_w = w // 2
        item_h = 16
        padding_x = 8
        start_y = y_offset + 4
        
        for i, key in enumerate(self.roi_keys):
            col = i % 2
            row = i // 2
            
            px = x_offset + padding_x + (col * col_w)
            py = start_y + (row * item_h)
            
            # Simpan area klik
            click_rect = QRectF(px, py, col_w - 5, item_h)
            self.legend_rects.append((click_rect, key))
            
            is_visible = self.visible_lines.get(key, True)
            
            # Draw color box
            box_color = QColor(self.roi_colors[i % len(self.roi_colors)])
            if not is_visible:
                box_color.setAlpha(50) # Redupkan
            painter.setBrush(box_color)
            painter.setPen(Qt.NoPen)
            painter.drawRect(px, py + 2, 8, 8)
            
            # Draw text
            text_color = QColor("#aaaaaa") if is_visible else QColor("#555555")
            painter.setPen(text_color)
            clean_name = key.replace("area_", "").replace("_", " ").title()
            painter.drawText(px + 14, py + 10, clean_name)

    def _draw_chart(self, painter, x_offset, y_offset, w, h_chart, title, data_lines, colors, max_y, visible_mask=None):
        padding = 4
        plot_h = h_chart - 22
        plot_w = w - (padding * 2)
        plot_x = padding
        plot_y = y_offset + 18
        
        # Judul
        painter.setPen(QColor("#cccccc"))
        font = painter.font()
        font.setPointSize(7)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(plot_x, y_offset + 12, title)
        
        # Background Grafik
        painter.fillRect(plot_x, plot_y, plot_w, plot_h, QColor("#1a1a1a"))
        
        # Kotak Garis Luar (Border)
        painter.setPen(QPen(QColor("#3a3a3a"), 1))
        painter.drawRect(plot_x, plot_y, plot_w, plot_h)
        
        # LOGIKA SLIDING WINDOW
        x_min = max(0, self.current_frame - (self.window_size // 2))
        x_max = x_min + self.window_size
        
        if x_max > self.total_frames:
            x_max = max(self.window_size, self.total_frames)
            x_min = max(0, x_max - self.window_size)
            
        window_range = x_max - x_min
        if window_range <= 0: return

        # Gambar Garis-Garis
        for i, line_data in enumerate(data_lines):
            if visible_mask is not None and not visible_mask[i]:
                continue
                
            # Gunakan ketebalan integer 1 (fast-path rendering) dan hindari RoundJoin
            pen = QPen(QColor(colors[i % len(colors)]), 1)
            painter.setPen(pen)
            
            slice_data = line_data[int(x_min):int(x_max)]
            if not slice_data: 
                continue
            
            poly = QPolygonF()
            for j, val in enumerate(slice_data):
                px = plot_x + (j / window_range) * plot_w
                py = plot_y + plot_h - (val / max_y) * plot_h
                poly.append(QPointF(px, py))
                
            painter.drawPolyline(poly)
            
        # Garis Cursor Tengah
        cursor_rel_idx = self.current_frame - x_min
        if 0 <= cursor_rel_idx <= window_range:
            cursor_px = plot_x + (cursor_rel_idx / window_range) * plot_w
            pen_cursor = QPen(QColor("#ffffff"), 1)
            pen_cursor.setStyle(Qt.DashLine)
            painter.setPen(pen_cursor)
            painter.drawLine(cursor_px, plot_y, cursor_px, plot_y + plot_h)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = event.position()
            for rect, key in self.legend_rects:
                if rect.contains(pos):
                    # Toggle visibility
                    self.visible_lines[key] = not self.visible_lines[key]
                    self.update()
                    return
        super().mousePressEvent(event)
