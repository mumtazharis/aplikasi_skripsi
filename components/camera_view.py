from PySide6.QtWidgets import QLabel, QSizePolicy
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage

class CameraView(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: black;")
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setMinimumSize(600, 450)
        self.current_pixmap = None

    def update_frame(self, image: QImage):
        """Menerima QImage dan mengubahnya menjadi QPixmap untuk ditampilkan."""
        self.current_pixmap = QPixmap.fromImage(image)
        self.update_scaled_pixmap()

    def update_scaled_pixmap(self):
        """Menyesuaikan ukuran gambar dengan ukuran window saat di-resize."""
        if not self.current_pixmap:
            return

        scaled = self.current_pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_scaled_pixmap()