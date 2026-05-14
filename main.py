import sys
import os
try:
    import pyi_splash
    pyi_splash.update_text("Initializing application...")
except ImportError:
    pass

os.environ["OPENCV_VIDEOIO_LOG_LEVEL"] = "0"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

if 'pyi_splash' in sys.modules:
    pyi_splash.update_text("Loading core system framework...")

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon
from PySide6.QtCore import Qt
from utils.resource_path import resource_path

# Fix missing mediapipe models path bug for PyInstaller
if getattr(sys, 'frozen', False):
    try:
        mp_path = os.path.join(sys._MEIPASS, 'mediapipe', 'modules')
        if os.path.exists(mp_path):
            os.environ['MEDIAPIPE_RESOURCE_DIR'] = mp_path
    except:
        pass

app = QApplication(sys.argv)
app.setWindowIcon(QIcon(resource_path("assets/icon.png")))

if 'pyi_splash' in sys.modules:
    pyi_splash.update_text("Initializing AI engines and models...")
    
from main_window import MainWindow
window = MainWindow()

window.showMaximized()
if 'pyi_splash' in sys.modules:
    pyi_splash.close()

window.setWindowState(window.windowState() | Qt.WindowActive)
window.raise_()
window.activateWindow()

app.exec()