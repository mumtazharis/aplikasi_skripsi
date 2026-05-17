from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QPainter, QColor, QPen, QPolygonF, QPixmap
from PySide6.QtCore import Qt, QRectF, QPointF

class RoiLineChartWidget(QWidget):
    """
    Widget untuk menggambar Grafik Energy ROI (Global dan Region).
    Menggunakan konsep Sliding Window layaknya Oscilloscope.

    OPTIMIZED: QPixmap double-buffering.
    Chart body di-render sekali ke _chart_pixmap saat window bergeser.
    Cursor update hanya blit pixmap + gambar garis cursor.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(220, 300)
        
        # Jendela frame yang divisualisasikan dalam satu waktu
        self.window_size = 150
        
        # Daftar ROI sama dengan yang diekstrak di CSV
        self.roi_keys = [
            "area_dahi", "area_alis_kanan", "area_alis_kiri", 
            "area_antara_alis", 
            "area_pipi_kanan", "area_pipi_kiri", "area_hidung", 
            "area_mulut_kanan", "area_mulut_kiri"
        ]
        
        # Warna-warni kontras tinggi — pre-cache QColor
        self._roi_color_hex = [
            "#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231", 
            "#911eb4", "#42d4f4", "#f032e6", "#bfef45"
        ]
        self.roi_colors = [QColor(c) for c in self._roi_color_hex]
        self._roi_colors_dim = []
        for c in self._roi_color_hex:
            qc = QColor(c)
            qc.setAlpha(50)
            self._roi_colors_dim.append(qc)
        
        self.history = {k: [] for k in self.roi_keys}
        self.global_history = []
        self.total_frames = 0
        self.current_frame = 0
        
        self.global_max_y = 0.001
        self.region_max_y = 0.001
        
        self.visible_lines = {k: True for k in self.roi_keys}
        self.legend_rects = []
        
        # --- QPixmap double-buffer ---
        self._chart_pixmap = None   # cached chart body (tanpa cursor)
        self._chart_dirty = True    # True = perlu rebuild
        self._cached_x_min = -1
        self._cached_x_max = -1
        
        # Pre-cached colors
        self._bg_color = QColor("#232323")
        self._plot_bg = QColor("#1a1a1a")
        self._border_color = QColor("#3a3a3a")
        self._cursor_color = QColor("#ffffff")
        self._text_color = QColor("#cccccc")
        self._no_data_color = QColor("#777777")
        self._global_line_color = QColor("#2d8ceb")
        self._legend_visible_color = QColor("#aaaaaa")
        self._legend_hidden_color = QColor("#555555")
        
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
        
        # Invalidate cache
        self._chart_dirty = True
        self._cached_x_min = -1
            
        self.update()

    def set_cursor(self, frame_idx):
        """
        Menggeser cursor window.
        """
        new_frame = max(0, min(frame_idx, self.total_frames - 1))
        if new_frame == self.current_frame:
            return
        self.current_frame = new_frame
        
        # Hitung apakah window bergeser
        x_min = max(0, self.current_frame - (self.window_size // 2))
        x_max = x_min + self.window_size
        if x_max > self.total_frames:
            x_max = max(self.window_size, self.total_frames)
            x_min = max(0, x_max - self.window_size)
        
        if x_min != self._cached_x_min or x_max != self._cached_x_max:
            self._chart_dirty = True   # window bergeser → rebuild chart
        
        self.update()

    def resizeEvent(self, event):
        """Invalidate cache saat ukuran widget berubah."""
        self._chart_dirty = True
        super().resizeEvent(event)

    # ===================================================================
    # PAINT — Double-buffered
    # ===================================================================

    def paintEvent(self, event):
        w = self.width()
        h = self.height()
        
        painter = QPainter(self)
        
        if self.total_frames == 0:
            painter.fillRect(0, 0, w, h, self._bg_color)
            painter.setPen(self._no_data_color)
            painter.drawText(self.rect(), Qt.AlignCenter, "No ROI Data")
            return
            
        # Potong area render jadi dua atas-bawah + ruang legend
        legend_h = 85
        available_h = h - legend_h
        global_h = available_h // 2
        region_h = available_h - global_h
        
        # Hitung window range
        x_min = max(0, self.current_frame - (self.window_size // 2))
        x_max = x_min + self.window_size
        if x_max > self.total_frames:
            x_max = max(self.window_size, self.total_frames)
            x_min = max(0, x_max - self.window_size)
        
        window_range = x_max - x_min
        if window_range <= 0:
            return
        
        # Langsung gambar ke widget tanpa QPixmap cache (lebih cepat karena bergeser tiap frame)
        painter.fillRect(0, 0, w, h, self._bg_color)
        
        padding = 4
        plot_w = w - (padding * 2)
        plot_x = padding
        
        # --- 1. GLOBAL CHART ---
        plot_h_g = global_h - 22
        plot_y_g = 18
        self._draw_chart_to_painter(
            painter, plot_x, 0, plot_w, global_h, plot_y_g, plot_h_g,
            "GLOBAL MOTION",
            [self.global_history[int(x_min):int(x_max)]],
            [self._global_line_color],
            self.global_max_y,
            [True],
            window_range
        )
        
        # --- 2. REGION CHART ---
        plot_h_r = region_h - 22
        plot_y_r = global_h + 18
        region_slices = [self.history[k][int(x_min):int(x_max)] for k in self.roi_keys]
        region_visible = [self.visible_lines[k] for k in self.roi_keys]
        self._draw_chart_to_painter(
            painter, plot_x, global_h, plot_w, region_h, plot_y_r, plot_h_r,
            "REGION MOTION",
            region_slices,
            self.roi_colors,
            self.region_max_y,
            region_visible,
            window_range
        )
        
        # --- 3. LEGEND ---
        self._draw_legend(painter, 0, h - legend_h, w, legend_h)
        
        # --- 4. DRAW CURSOR ---
        cursor_rel_idx = self.current_frame - x_min
        if 0 <= cursor_rel_idx <= window_range:
            cursor_px = plot_x + (cursor_rel_idx / window_range) * plot_w
            pen_cursor = QPen(self._cursor_color, 1)
            pen_cursor.setStyle(Qt.DashLine)
            painter.setPen(pen_cursor)
            
            painter.drawLine(int(cursor_px), plot_y_g, int(cursor_px), plot_y_g + plot_h_g)
            painter.drawLine(int(cursor_px), plot_y_r, int(cursor_px), plot_y_r + plot_h_r)

    def _draw_chart_to_painter(self, painter, plot_x, y_offset, plot_w, h_chart, 
                                plot_y, plot_h, title, data_slices, colors, max_y, 
                                visible_mask, window_range):
        """Render satu chart (global/region) ke painter."""
        # Judul
        painter.setPen(self._text_color)
        font = painter.font()
        font.setPointSize(7)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(plot_x, y_offset + 12, title)
        
        # Background Grafik
        painter.fillRect(plot_x, plot_y, plot_w, plot_h, self._plot_bg)
        
        # Border
        painter.setPen(QPen(self._border_color, 1))
        painter.drawRect(plot_x, plot_y, plot_w, plot_h)
        
        if window_range <= 0:
            return
        
        # Gambar garis
        for i, slice_data in enumerate(data_slices):
            if not slice_data:
                continue
            if visible_mask is not None and not visible_mask[i]:
                continue
            
            pen = QPen(colors[i % len(colors)], 1)
            painter.setPen(pen)
            
            poly = QPolygonF()
            for j, val in enumerate(slice_data):
                px = plot_x + (j / window_range) * plot_w
                py = plot_y + plot_h - (val / max_y) * plot_h
                poly.append(QPointF(px, py))
            
            painter.drawPolyline(poly)
        
    def _draw_legend(self, painter, x_offset, y_offset, w, h_legend):
        self.legend_rects.clear()
        painter.setPen(self._text_color)
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
            
            # Draw color box — gunakan pre-cached color
            if is_visible:
                box_color = self.roi_colors[i % len(self.roi_colors)]
            else:
                box_color = self._roi_colors_dim[i % len(self._roi_colors_dim)]
            painter.setBrush(box_color)
            painter.setPen(Qt.NoPen)
            painter.drawRect(px, py + 2, 8, 8)
            
            # Draw text
            text_color = self._legend_visible_color if is_visible else self._legend_hidden_color
            painter.setPen(text_color)
            clean_name = key.replace("area_", "").replace("_", " ").title()
            painter.drawText(px + 14, py + 10, clean_name)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = event.position()
            for rect, key in self.legend_rects:
                if rect.contains(pos):
                    # Toggle visibility
                    self.visible_lines[key] = not self.visible_lines[key]
                    # Invalidate cache
                    self._chart_dirty = True
                    self.update()
                    return
        super().mousePressEvent(event)
