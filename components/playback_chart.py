from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QPointF, Signal
from PySide6.QtGui import (
    QPainter, QColor, QPen,
    QPolygonF, QFont, QFontMetrics,
    QPixmap
)


class PlaybackChart(QWidget):
    """
    Dual timeline chart: Macro (top) and Micro (bottom).
    Both rendered as identical line charts.
    Adobe Premiere Pro dark theme.

    OPTIMIZED: QPixmap double-buffering.
    Chart body di-render sekali ke _chart_pixmap.
    Cursor update hanya blit pixmap + gambar garis cursor.
    """
    seek_requested = Signal(int)

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(160)
        self.setCursor(Qt.PointingHandCursor)

        self.macro_data = []    # list of (score, color_key)
        self.micro_data = []    # list of (score, color_key)
        self.cursor_pos = 0
        self.total_frames = 0

        self.zoom_factor = 2.0
        self.base_width = 800
        self.is_dragging = False

        # --- QPixmap cache ---
        self._chart_pixmap = None   # cached chart body (tanpa cursor)
        self._chart_dirty = True    # True = perlu rebuild pixmap

        # Pre-cached QColor objects
        self._color_cache = {
            "#4caf50": QColor("#4caf50"),
            "#e74c3c": QColor("#e74c3c"),
            "#e6a817": QColor("#e6a817"),
            "#555555": QColor("#555555"),
        }
        # Pre-cached fill colors (alpha=30)
        self._fill_cache = {}
        for key, col in self._color_cache.items():
            fc = QColor(col)
            fc.setAlpha(30)
            self._fill_cache[key] = fc

        # Pre-cached fonts
        self._label_font = QFont("Segoe UI", 7, QFont.Bold)
        self._small_font = QFont("Consolas", 6)
        self._frame_font = QFont("Consolas", 7)
        self._empty_font = QFont("Segoe UI", 10)

        # Pre-cached colors
        self._bg_color = QColor("#232323")
        self._grid_color = QColor("#2e2e2e")
        self._cursor_color = QColor("#2d8ceb")
        self._divider_color = QColor("#3a3a3a")
        self._text_dim = QColor("#555555")
        self._text_axis = QColor("#444444")
        self._text_label = QColor("#666666")
        self._text_bright = QColor("#cccccc")

    def set_data(self, prediction_data):
        self.macro_data = []
        self.micro_data = []

        for row in prediction_data:
            # Macro
            label = row["macro_label"].lower()
            conf = row["macro_confidence"]

            if label == "positive":
                score = conf * 100
                color_key = "#4caf50"
            elif label == "negative":
                score = -conf * 100
                color_key = "#e74c3c"
            elif label == "neutral":
                score = 0
                color_key = "#e6a817"
            else:
                score = 0
                color_key = "#555555"
            self.macro_data.append((score, color_key))

            # Micro
            micro_label = row.get("micro_label", "").strip().lower()
            micro_conf = row.get("micro_confidence", 0)
            if isinstance(micro_conf, str):
                try:
                    micro_conf = float(micro_conf)
                except ValueError:
                    micro_conf = 0

            if micro_label == "positive":
                micro_score = micro_conf * 100
                micro_color_key = "#4caf50"
            elif micro_label == "negative":
                micro_score = -micro_conf * 100
                micro_color_key = "#e74c3c"
            else:
                micro_score = 0
                micro_color_key = "#555555"
            self.micro_data.append((micro_score, micro_color_key))

        self.total_frames = len(self.macro_data)
        self.cursor_pos = 0
        self._chart_dirty = True   # <<< invalidate cache
        self._update_chart_width()

    def _update_chart_width(self):
        if self.total_frames > 0:
            new_width = max(self.base_width, int(self.total_frames * self.zoom_factor))
            self.setMinimumWidth(new_width)
            self.resize(new_width, self.height())
        self._chart_dirty = True   # <<< invalidate cache
        self.update()
    
    def wheelEvent(self, event):
        if event.modifiers() == Qt.ControlModifier:
            delta = event.angleDelta().y()
            if delta > 0:
                self.zoom_factor *= 1.2
            else:
                self.zoom_factor /= 1.2

            self.zoom_factor = max(0.1, min(self.zoom_factor, 50.0))
            self._update_chart_width()
        else:
            super().wheelEvent(event)

    def resizeEvent(self, event):
        """Invalidate cache saat ukuran widget berubah."""
        self._chart_dirty = True
        super().resizeEvent(event)

    def set_cursor(self, frame_idx):
        if frame_idx == self.cursor_pos:
            return
        self.cursor_pos = frame_idx
        self.update()   # hanya blit cache + cursor line

    def clear_data(self):
        self.macro_data = []
        self.micro_data = []
        self.total_frames = 0
        self.cursor_pos = 0
        self._chart_dirty = True
        self.update()

    def _update_seek_position(self, x_pos):
        if self.total_frames == 0:
            return
            
        w = self.width()
        frame_idx = int((x_pos / w) * self.total_frames)
        frame_idx = max(0, min(frame_idx, self.total_frames - 1))
        
        if self.cursor_pos != frame_idx:
            self.cursor_pos = frame_idx
            self.seek_requested.emit(frame_idx)
            self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.is_dragging = True
            self._update_seek_position(event.position().x())

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self._update_seek_position(event.position().x())
   
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.is_dragging = False

    # ===================================================================
    # PAINT — Double-buffered: blit cached pixmap + draw cursor on top
    # ===================================================================

    def paintEvent(self, event):
        w = self.width()
        h = self.height()

        # Rebuild chart body pixmap if dirty
        if self._chart_dirty or self._chart_pixmap is None or self._chart_pixmap.size() != self.size():
            self._rebuild_chart_pixmap(w, h)
            self._chart_dirty = False

        painter = QPainter(self)

        # 1. Blit cached chart body HANYA pada area yang butuh di-repaint (sangat cepat)
        rect = event.rect()
        painter.drawPixmap(rect.topLeft(), self._chart_pixmap, rect)

        # 2. Draw cursor on top (ringan — hanya 1 garis + segitiga + teks)
        if self.macro_data and self.total_frames > 0:
            self._draw_cursor(painter, w, h)

    def _rebuild_chart_pixmap(self, w, h):
        """Render seluruh chart body ke QPixmap cache. Dipanggil SEKALI saat data/zoom/size berubah."""
        self._chart_pixmap = QPixmap(w, h)
        painter = QPainter(self._chart_pixmap)

        # Background
        painter.fillRect(0, 0, w, h, self._bg_color)

        if not self.macro_data:
            painter.setPen(self._text_dim)
            painter.setFont(self._empty_font)
            painter.drawText(self.rect(), Qt.AlignCenter, "No prediction data")
            painter.end()
            return

        total = len(self.macro_data)
        divider_y = h // 2
        macro_h = divider_y
        micro_h = h - divider_y

        # Macro chart
        painter.save()
        painter.setClipRect(0, 0, w, macro_h)
        self._draw_line_chart(painter, self.macro_data, 0, macro_h, w, total)
        painter.restore()

        # Row label
        painter.setFont(self._label_font)
        painter.setPen(self._text_label)
        painter.drawText(6, 11, "MACRO")

        # Divider
        painter.setPen(QPen(self._divider_color, 1))
        painter.drawLine(0, divider_y, w, divider_y)

        # Micro chart
        painter.save()
        painter.setClipRect(0, divider_y, w, micro_h)
        self._draw_line_chart(painter, self.micro_data, divider_y, micro_h, w, total)
        painter.restore()

        # Row label
        painter.setPen(self._text_label)
        painter.setFont(self._label_font)
        painter.drawText(6, divider_y + 11, "MICRO")

        painter.end()

    def _draw_cursor(self, painter, w, h):
        """Gambar cursor line + triangle + label. Sangat ringan."""
        total = len(self.macro_data)
        cursor_x = (self.cursor_pos / total) * w

        # Cursor line
        pen_cursor = QPen(self._cursor_color, 1)
        painter.setPen(pen_cursor)
        painter.drawLine(QPointF(cursor_x, 0), QPointF(cursor_x, h))

        # Triangle
        painter.setBrush(self._cursor_color)
        painter.setPen(Qt.NoPen)
        tri = QPolygonF()
        tri.append(QPointF(cursor_x - 4, 0))
        tri.append(QPointF(cursor_x + 4, 0))
        tri.append(QPointF(cursor_x, 6))
        painter.drawPolygon(tri)

        # Frame label
        painter.setPen(self._text_bright)
        painter.setFont(self._frame_font)
        label_text = f"F:{self.cursor_pos}"
        fm = QFontMetrics(self._frame_font)
        text_w = fm.horizontalAdvance(label_text)

        label_x = cursor_x + 6
        if label_x + text_w > w:
            label_x = cursor_x - text_w - 6
        painter.drawText(int(label_x), 15, label_text)

    # ===================================================================
    # LINE CHART RENDERER — dipanggil hanya saat rebuild pixmap
    # ===================================================================

    def _draw_line_chart(self, painter, data, y_offset, chart_h, w, total):
        """Render line chart ke painter. Hanya dipanggil saat _rebuild_chart_pixmap."""
        painter.fillRect(0, y_offset, w, chart_h, self._bg_color)

        # Grid
        pen_grid = QPen(self._grid_color, 1)
        pen_grid.setStyle(Qt.DashLine)
        painter.setPen(pen_grid)

        y_zero = y_offset + chart_h / 2
        y_top = y_offset + chart_h * 0.25
        y_bot = y_offset + chart_h * 0.75

        painter.drawLine(0, int(y_zero), w, int(y_zero))
        painter.drawLine(0, int(y_top), w, int(y_top))
        painter.drawLine(0, int(y_bot), w, int(y_bot))

        # Axis labels
        painter.setPen(self._text_axis)
        painter.setFont(self._small_font)
        painter.drawText(w - 20, int(y_top) + 9, "+50")
        painter.drawText(w - 10, int(y_zero) + 9, "0")
        painter.drawText(w - 20, int(y_bot) + 9, "-50")

        inv_200 = 1.0 / 200.0

        def get_y(val):
            normalized = (val + 100) * inv_200
            return y_offset + chart_h - (normalized * chart_h)

        step_x = w / total if total > 0 else 0

        if total > w * 2:
            # Downsampled mode
            inv_w = 1.0 / w if w > 0 else 0
            for px in range(int(w)):
                start_idx = int(px * inv_w * total)
                end_idx = min(int((px + 1) * inv_w * total), total)
                if start_idx >= total:
                    break

                segment = data[start_idx:end_idx]
                if not segment:
                    continue

                avg_score = sum(s for s, _ in segment) / len(segment)
                last_color_key = segment[-1][1]
                y_val = get_y(avg_score)

                fill_col = self._fill_cache.get(last_color_key)
                if fill_col:
                    painter.setPen(Qt.NoPen)
                    painter.setBrush(fill_col)
                    painter.drawRect(px, int(y_val), 1, int(y_offset + chart_h - y_val))

                col = self._color_cache.get(last_color_key)
                if col:
                    painter.setBrush(col)
                    painter.drawRect(px, int(y_val), 1, 2)
        else:
            # Full resolution mode
            i = 0
            while i < total - 1:
                color_key = data[i + 1][1]
                col = self._color_cache.get(color_key, self._text_dim)
                fill_col = self._fill_cache.get(color_key)

                val1, _ = data[i]
                val2, _ = data[i + 1]
                x1 = i * step_x
                y1 = get_y(val1)
                x2 = (i + 1) * step_x
                y2 = get_y(val2)

                # Fill polygon
                if fill_col:
                    polygon = QPolygonF()
                    polygon.append(QPointF(x1, y1))
                    polygon.append(QPointF(x2, y2))
                    polygon.append(QPointF(x2, y_offset + chart_h))
                    polygon.append(QPointF(x1, y_offset + chart_h))
                    painter.setPen(Qt.NoPen)
                    painter.setBrush(fill_col)
                    painter.drawPolygon(polygon)

                # Line
                pen = QPen(col, 2)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))

                i += 1
