from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QPointF, Signal
from PySide6.QtGui import (
    QPainter, QColor, QPen,
    QPolygonF, QFont, QFontMetrics
)


class PlaybackChart(QWidget):
    """
    Dual timeline chart: Macro (top) and Micro (bottom).
    Both rendered as identical line charts.
    Adobe Premiere Pro dark theme.
    """
    seek_requested = Signal(int)

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(160)
        self.setCursor(Qt.PointingHandCursor)

        self.macro_data = []
        self.micro_data = []
        self.cursor_pos = 0
        self.total_frames = 0

        self.zoom_factor = 2.0
        self.base_width = 800
        self.is_dragging = False

    def set_data(self, prediction_data):
        self.macro_data = []
        self.micro_data = []

        for row in prediction_data:
            # Macro
            label = row["macro_label"].lower()
            conf = row["macro_confidence"]

            if label == "positive":
                score = conf * 100
                color = "#4caf50"
            elif label == "negative":
                score = -conf * 100
                color = "#e74c3c"
            elif label == "neutral":
                score = 0
                color = "#e6a817"
            else:
                score = 0
                color = "#555555"
            self.macro_data.append((score, color))

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
                micro_color = "#4caf50"
            elif micro_label == "negative":
                micro_score = -micro_conf * 100
                micro_color = "#e74c3c"
            else:
                micro_score = 0
                micro_color = "#555555"
            self.micro_data.append((micro_score, micro_color))

        self.total_frames = len(self.macro_data)
        self.cursor_pos = 0
        self._update_chart_width()

    def _update_chart_width(self):
        if self.total_frames > 0:
            # Lebar baru = total frame dikali zoom (semakin besar zoom, semakin panjang chart-nya)
            new_width = max(self.base_width, int(self.total_frames * self.zoom_factor))
            self.setMinimumWidth(new_width)
            self.resize(new_width, self.height())
        self.update()
    
    def wheelEvent(self, event):
        if event.modifiers() == Qt.ControlModifier:
            # Jika menahan tombol CTRL sambil scroll
            delta = event.angleDelta().y()
            if delta > 0:
                self.zoom_factor *= 1.2 # Zoom in (renggang)
            else:
                self.zoom_factor /= 1.2 # Zoom out (dempet)

            # Batasi maksimal dan minimal zoom
            self.zoom_factor = max(0.1, min(self.zoom_factor, 50.0))
            self._update_chart_width()
        else:
            # Scroll biasa diteruskan ke QScrollArea parent
            super().wheelEvent(event)

    def set_cursor(self, frame_idx):
        self.cursor_pos = frame_idx
        self.update()

    def clear_data(self):
        self.macro_data = []
        self.micro_data = []
        self.total_frames = 0
        self.cursor_pos = 0
        self.update()

    def _update_seek_position(self, x_pos):
        if self.total_frames == 0:
            return
            
        w = self.width()
        # Hitung index frame berdasarkan posisi klik/drag
        frame_idx = int((x_pos / w) * self.total_frames)
        
        # Pastikan index tidak keluar dari batas (0 sampai total_frames - 1)
        frame_idx = max(0, min(frame_idx, self.total_frames - 1))
        
        # Optimasi: Hanya update dan emit signal jika posisinya benar-benar berubah
        if self.cursor_pos != frame_idx:
            self.cursor_pos = frame_idx
            self.seek_requested.emit(frame_idx)
            self.update()

    def mousePressEvent(self, event):
        # Hanya respon klik kiri
        if event.button() == Qt.LeftButton:
            self.is_dragging = True
            self._update_seek_position(event.position().x())

    def mouseMoveEvent(self, event):
        # Hanya respon saat mouse digeser DAN klik kiri sedang ditahan
        if event.buttons() & Qt.LeftButton:
            self._update_seek_position(event.position().x())
   
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.is_dragging = False
   
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()

        # Adobe dark background
        painter.fillRect(0, 0, w, h, QColor("#232323"))

        if not self.macro_data:
            painter.setPen(QColor("#555555"))
            font = QFont("Segoe UI", 10)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignCenter, "No prediction data")
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
        label_font = QFont("Segoe UI", 7, QFont.Bold)
        painter.setFont(label_font)
        painter.setPen(QColor("#666666"))
        painter.drawText(6, 11, "MACRO")

        # Divider
        painter.setPen(QPen(QColor("#3a3a3a"), 1))
        painter.drawLine(0, divider_y, w, divider_y)

        # Micro chart
        painter.save()
        painter.setClipRect(0, divider_y, w, micro_h)
        self._draw_line_chart(painter, self.micro_data, divider_y, micro_h, w, total)
        painter.restore()

        # Row label
        painter.setPen(QColor("#666666"))
        painter.setFont(label_font)
        painter.drawText(6, divider_y + 11, "MICRO")

        # Cursor
        if total > 0:
            cursor_x = (self.cursor_pos / total) * w

            # Cursor line — Adobe blue
            pen_cursor = QPen(QColor("#2d8ceb"), 1)
            painter.setPen(pen_cursor)
            painter.drawLine(QPointF(cursor_x, 0), QPointF(cursor_x, h))

            # Triangle
            painter.setBrush(QColor("#2d8ceb"))
            painter.setPen(Qt.NoPen)
            tri = QPolygonF()
            tri.append(QPointF(cursor_x - 4, 0))
            tri.append(QPointF(cursor_x + 4, 0))
            tri.append(QPointF(cursor_x, 6))
            painter.drawPolygon(tri)

            # Frame label
            painter.setPen(QColor("#cccccc"))
            fl = QFont("Consolas", 7)
            painter.setFont(fl)
            label_text = f"F:{self.cursor_pos}"
            fm = QFontMetrics(fl)
            text_w = fm.horizontalAdvance(label_text)

            label_x = cursor_x + 6
            if label_x + text_w > w:
                label_x = cursor_x - text_w - 6
            painter.drawText(int(label_x), 15, label_text)

    def _draw_line_chart(self, painter, data, y_offset, chart_h, w, total):
        """Identical line chart renderer for both macro and micro."""
        painter.fillRect(0, y_offset, w, chart_h, QColor("#232323"))

        # Grid
        pen_grid = QPen(QColor("#2e2e2e"), 1)
        pen_grid.setStyle(Qt.DashLine)
        painter.setPen(pen_grid)

        y_zero = y_offset + chart_h / 2
        y_top = y_offset + chart_h * 0.25
        y_bot = y_offset + chart_h * 0.75

        painter.drawLine(0, int(y_zero), w, int(y_zero))
        painter.drawLine(0, int(y_top), w, int(y_top))
        painter.drawLine(0, int(y_bot), w, int(y_bot))

        # Axis labels
        painter.setPen(QColor("#444444"))
        small_font = QFont("Consolas", 6)
        painter.setFont(small_font)
        painter.drawText(w - 20, int(y_top) + 9, "+50")
        painter.drawText(w - 10, int(y_zero) + 9, "0")
        painter.drawText(w - 20, int(y_bot) + 9, "-50")

        def get_y(val):
            normalized = (val + 100) / 200
            return y_offset + chart_h - (normalized * chart_h)

        step_x = w / total if total > 0 else 0

        if total > w * 2:
            for px in range(int(w)):
                start_idx = int((px / w) * total)
                end_idx = min(int(((px + 1) / w) * total), total)
                if start_idx >= total:
                    break

                segment = data[start_idx:end_idx]
                if not segment:
                    continue

                avg_score = sum(s for s, _ in segment) / len(segment)
                last_color = segment[-1][1]
                y_val = get_y(avg_score)
                col = QColor(last_color)

                fill_col = QColor(col)
                fill_col.setAlpha(30)
                painter.setPen(Qt.NoPen)
                painter.setBrush(fill_col)
                painter.drawRect(px, int(y_val), 1, int(y_offset + chart_h - y_val))

                painter.setBrush(col)
                painter.drawRect(px, int(y_val), 1, 2)
        else:
            for i in range(total - 1):
                val1, col1 = data[i]
                val2, col_hex = data[i + 1]

                x1 = i * step_x
                y1 = get_y(val1)
                x2 = (i + 1) * step_x
                y2 = get_y(val2)

                col = QColor(col_hex)

                polygon = QPolygonF()
                polygon.append(QPointF(x1, y1))
                polygon.append(QPointF(x2, y2))
                polygon.append(QPointF(x2, y_offset + chart_h))
                polygon.append(QPointF(x1, y_offset + chart_h))

                fill_color = QColor(col)
                fill_color.setAlpha(30)
                painter.setPen(Qt.NoPen)
                painter.setBrush(fill_color)
                painter.drawPolygon(polygon)

                pen = QPen(col, 2)
                pen.setCapStyle(Qt.RoundCap)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))
