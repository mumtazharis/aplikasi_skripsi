from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QStackedWidget,
    QPushButton, QFrame, QLabel
)
from PySide6.QtCore import Qt

from components.input_page import InputPage
from components.dashboard_page import DashboardPage
from styles import NAV_BAR_STYLE, NAV_BUTTON_STYLE, NAV_BUTTON_ACTIVE_STYLE, MAIN_WINDOW_STYLE


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Expression Analyzer")
        self.setObjectName("mainWindow")
        self.resize(1200, 700)
        self.setStyleSheet(MAIN_WINDOW_STYLE)

        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ====== NAVIGATION BAR ======
        nav_bar = QFrame()
        nav_bar.setObjectName("navBar")
        nav_bar.setFixedHeight(44)
        nav_bar.setStyleSheet(NAV_BAR_STYLE)

        nav_layout = QHBoxLayout(nav_bar)
        nav_layout.setContentsMargins(16, 0, 16, 0)
        nav_layout.setSpacing(0)

        # App title
        app_title = QLabel("EXPRESSION ANALYZER")
        app_title.setObjectName("appTitle")
        nav_layout.addWidget(app_title)

        nav_layout.addSpacing(30)

        # Navigation buttons (English, no emoji)
        self.btn_input = QPushButton("INPUT & PREDICT")
        self.btn_input.setFixedHeight(44)
        self.btn_input.setCursor(Qt.PointingHandCursor)
        self.btn_input.clicked.connect(lambda: self.switch_page(0))
        nav_layout.addWidget(self.btn_input)

        self.btn_dashboard = QPushButton("ANALYSIS")
        self.btn_dashboard.setFixedHeight(44)
        self.btn_dashboard.setCursor(Qt.PointingHandCursor)
        self.btn_dashboard.clicked.connect(lambda: self.switch_page(1))
        nav_layout.addWidget(self.btn_dashboard)

        nav_layout.addStretch()

        main_layout.addWidget(nav_bar)

        # ====== PAGE STACK ======
        self.page_stack = QStackedWidget()

        # Page 0: Input & Predict
        self.input_page = InputPage()
        self.input_page.prediction_finished.connect(self._on_prediction_finished)
        self.input_page.btn_open_dashboard.clicked.connect(self._open_dashboard_with_result)
        self.page_stack.addWidget(self.input_page)

        # Page 1: Analysis Dashboard
        self.dashboard_page = DashboardPage()
        self.page_stack.addWidget(self.dashboard_page)

        main_layout.addWidget(self.page_stack)

        # Set initial page
        self.current_page = 0
        self.switch_page(0)

    def switch_page(self, index):
        self.current_page = index
        self.page_stack.setCurrentIndex(index)
        self._update_nav_styles()

    def _update_nav_styles(self):
        if self.current_page == 0:
            self.btn_input.setStyleSheet(NAV_BUTTON_ACTIVE_STYLE)
            self.btn_dashboard.setStyleSheet(NAV_BUTTON_STYLE)
        else:
            self.btn_input.setStyleSheet(NAV_BUTTON_STYLE)
            self.btn_dashboard.setStyleSheet(NAV_BUTTON_ACTIVE_STYLE)

    def _on_prediction_finished(self, csv_path):
        self.last_csv_path = csv_path

    def _open_dashboard_with_result(self):
        if hasattr(self, 'last_csv_path') and self.last_csv_path:
            self.switch_page(1)
            self.dashboard_page.load_csv(self.last_csv_path)

    def closeEvent(self, event):
        self.input_page.cleanup()
        self.dashboard_page.cleanup()
        event.accept()