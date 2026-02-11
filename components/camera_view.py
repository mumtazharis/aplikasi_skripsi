from PySide6.QtWidgets import QLabel, QSizePolicy
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen, QColor

class CameraView(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: black;")
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setMinimumSize(600, 450)
        self.current_pixmap = None
        self.face_rect = None
    
    def update_face_box(self, x1, y1, x2, y2):
        """
        Slot ini dipanggil oleh MLWorker.
        Hanya menyimpan koordinat, tidak langsung menggambar.
        Jika x1=0, y1=0, dll, artinya wajah hilang.
        """
        if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
            self.face_rect = None
        else:
            self.face_rect = (x1, y1, x2, y2)

    def update_frame(self, image: QImage):
        """Menerima QImage, menggambar kotak (jika ada), lalu menampilkan."""
        
        # 1. Convert QImage ke QPixmap (Canvas kita)
        pixmap = QPixmap.fromImage(image)

        # 2. Cek apakah ada koordinat wajah yang tersimpan
        if self.face_rect:
            painter = QPainter(pixmap)
            
            # Setup Pen (Warna Garis)
            pen = QPen(QColor(0, 255, 0)) # Warna Hijau
            pen.setWidth(1)               # Ketebalan garis
            painter.setPen(pen)

            # Gambar Kotak
            x1, y1, x2, y2 = self.face_rect
            w_box = x2 - x1
            h_box = y2 - y1
            
            # DrawRect butuh (x, y, width, height)
            painter.drawRect(x1, y1, w_box, h_box)
            
            painter.end() # Selesai menggambar

        # 3. Simpan sebagai pixmap saat ini
        self.current_pixmap = pixmap
        
        # 4. Tampilkan (dengan scaling)
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