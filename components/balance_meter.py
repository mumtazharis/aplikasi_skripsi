from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QColor, QPen, QBrush


class BalanceMeter(QWidget):
    def __init__(self):
        super().__init__()
        self.value = 0  # range -100 sampai 100
        self.setMinimumHeight(5)

    def setValue(self, value):
        self.value = max(-100, min(100, value))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        width = self.width()
        height = self.height()
        center_x = width // 2
        radius = height / 2

        # Background rounded
        painter.setBrush(QColor("#1e1e1e"))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(0, 0, width, height, radius, radius)

        # Tentukan warna
        if self.value < -30:
            color = QColor("#ff4c4c")
        elif self.value > 30:
            color = QColor("#00d27a")
        else:
            color = QColor("#ffaa33")

        painter.setBrush(QBrush(color))

        max_half_width = width // 2
        bar_length = int((abs(self.value) / 100) * max_half_width)

        if self.value >= 0:
            painter.drawRoundedRect(center_x, 0, bar_length, height, radius, radius)
        else:
            painter.drawRoundedRect(center_x - bar_length, 0, bar_length, height, radius, radius)

        # Center line
        painter.setPen(QPen(QColor("#ffffff"), 2))
        painter.drawLine(center_x, 4, center_x, height - 4)
