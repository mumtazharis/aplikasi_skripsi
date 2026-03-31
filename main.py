import sys
from PySide6.QtWidgets import QApplication
from main_window import MainWindow
from PySide6.QtGui import QIcon
from PySide6.QtCore import Qt
from utils.resource_path import resource_path
try:
    import pyi_splash
except ImportError:
    pass

app = QApplication(sys.argv)
app.setWindowIcon(QIcon(resource_path("assets/icon.png")))

window = MainWindow()

if 'pyi_splash' in sys.modules:
        pyi_splash.close()

window.showMaximized()

# Memaksa window ke lapisan paling atas (di atas Explorer/Taskbar)
window.setWindowFlags(window.windowFlags() | Qt.WindowStaysOnTopHint)
window.showMaximized() 

# Hapus flag "Always on Top" agar user bisa buka aplikasi lain di atasnya nanti
window.setWindowFlags(window.windowFlags() & ~Qt.WindowStaysOnTopHint)
window.showMaximized()

window.raise_()     
window.activateWindow() 
app.exec()
