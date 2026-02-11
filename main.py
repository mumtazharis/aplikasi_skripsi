import sys
from PySide6.QtWidgets import QApplication
from main_window import MainWindow
from PySide6.QtGui import QIcon

app = QApplication(sys.argv)
app.setWindowIcon(QIcon("assets/icon.png"))
window = MainWindow()
window.showMaximized()
app.exec()
