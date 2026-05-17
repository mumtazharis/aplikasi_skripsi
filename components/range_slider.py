from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QPainter, QColor, QPen, QBrush
from PySide6.QtCore import Qt, Signal, QRect

class RangeSlider(QWidget):
    rangeChanged = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(40)
        self.setMinimumWidth(150)
        self._minimum = 0
        self._maximum = 100
        self._low = 0
        self._high = 100
        self._active_handle = 0 # 0: none, 1: low, 2: high
        self._handle_radius = 8

    def setMinimum(self, val):
        self._minimum = val
        self.update()

    def setMaximum(self, val):
        self._maximum = val
        self.update()

    def setLow(self, val):
        self._low = max(self._minimum, min(val, self._high))
        self.update()

    def setHigh(self, val):
        self._high = min(self._maximum, max(val, self._low))
        self.update()

    def getRange(self):
        return self._low, self._high

    def _get_handle_rects(self):
        w = self.width() - self._handle_radius * 2
        h = self.height()
        range_span = max(1, self._maximum - self._minimum)
        
        low_x = self._handle_radius + (self._low - self._minimum) / range_span * w
        high_x = self._handle_radius + (self._high - self._minimum) / range_span * w
        
        y = h / 2
        r = self._handle_radius
        
        rect_low = QRect(int(low_x - r), int(y - r), int(r*2), int(r*2))
        rect_high = QRect(int(high_x - r), int(y - r), int(r*2), int(r*2))
        return rect_low, rect_high, y

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect_low, rect_high, cy = self._get_handle_rects()
        
        is_enabled = self.isEnabled()
        
        # Colors
        track_color = QColor("#333333") if is_enabled else QColor("#222222")
        active_track_color = QColor("#0e639c") if is_enabled else QColor("#334455")
        handle_color = QColor("#ffffff") if is_enabled else QColor("#888888")
        handle_border_active = QColor("#4aa3f5") if is_enabled else QColor("#555555")

        # Draw track
        track_height = 4
        track_rect = QRect(self._handle_radius, int(cy - track_height/2), self.width() - self._handle_radius*2, track_height)
        painter.setPen(Qt.NoPen)
        painter.setBrush(track_color)
        painter.drawRoundedRect(track_rect, 2, 2)

        # Draw active track
        active_rect = QRect(rect_low.center().x(), int(cy - track_height/2), rect_high.center().x() - rect_low.center().x(), track_height)
        painter.setBrush(active_track_color)
        painter.drawRoundedRect(active_rect, 2, 2)

        # Draw handles
        painter.setBrush(handle_color)
        if self._active_handle == 1 and is_enabled:
            painter.setPen(QPen(handle_border_active, 2))
        else:
            painter.setPen(Qt.NoPen)
        painter.drawEllipse(rect_low)

        if self._active_handle == 2 and is_enabled:
            painter.setPen(QPen(handle_border_active, 2))
        else:
            painter.setPen(Qt.NoPen)
        painter.drawEllipse(rect_high)

    def mousePressEvent(self, event):
        rect_low, rect_high, _ = self._get_handle_rects()
        # Expand hit area slightly for easier grabbing
        hit_margin = 4
        rect_low_hit = rect_low.adjusted(-hit_margin, -hit_margin, hit_margin, hit_margin)
        rect_high_hit = rect_high.adjusted(-hit_margin, -hit_margin, hit_margin, hit_margin)
        
        # Prefer high handle if they overlap and we click the right side, or prefer active one
        if rect_low_hit.contains(event.pos()) and rect_high_hit.contains(event.pos()):
            # If handles are very close, decide based on which side of the center we clicked
            center_x = (rect_low.center().x() + rect_high.center().x()) / 2
            if event.pos().x() < center_x:
                self._active_handle = 1
            else:
                self._active_handle = 2
        elif rect_low_hit.contains(event.pos()):
            self._active_handle = 1
        elif rect_high_hit.contains(event.pos()):
            self._active_handle = 2
        else:
            self._active_handle = 0
            
        self.update()

    def mouseMoveEvent(self, event):
        if self._active_handle == 0:
            return
            
        w = self.width() - self._handle_radius * 2
        range_span = max(1, self._maximum - self._minimum)
        
        # Calculate new value
        x = max(self._handle_radius, min(event.pos().x(), self.width() - self._handle_radius))
        val = self._minimum + (x - self._handle_radius) / w * range_span
        val = int(round(val))
        
        if self._active_handle == 1:
            self._low = max(self._minimum, min(val, self._high))
        elif self._active_handle == 2:
            self._high = min(self._maximum, max(val, self._low))
            
        self.update()
        self.rangeChanged.emit(self._low, self._high)

    def mouseReleaseEvent(self, event):
        self._active_handle = 0
        self.update()
