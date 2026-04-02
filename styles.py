"""
Centralized styles for the application.
Adobe Premiere Pro / Photoshop inspired dark theme.
"""

# Adobe-style color palette
COLORS = {
    "bg_darkest": "#1a1a1a",       # Deepest background (canvas)
    "bg_dark": "#232323",           # Main panel background
    "bg_panel": "#2d2d2d",          # Sidebar / elevated panels
    "bg_card": "#383838",           # Cards / input fields
    "bg_header": "#1e1e1e",         # Top bar / headers
    "bg_hover": "#404040",          # Hover state
    "border": "#3a3a3a",            # Default border
    "border_light": "#4a4a4a",      # Light border / dividers
    "text_primary": "#e0e0e0",      # Primary text
    "text_secondary": "#999999",    # Secondary text
    "text_muted": "#666666",        # Muted / hint text
    "accent": "#2d8ceb",            # Primary accent (Adobe blue)
    "accent_hover": "#4aa3f5",      # Accent hover
    "accent_dim": "#1a5a9e",        # Accent disabled / dim
    "positive": "#4caf50",          # Green
    "neutral": "#e6a817",           # Amber/Yellow
    "negative": "#e74c3c",          # Red
    "danger": "#c0392b",            # Deep red
    "danger_hover": "#e04838",      # Danger hover
}

# Main window background
MAIN_WINDOW_STYLE = """
    QWidget#mainWindow {
        background-color: #1a1a1a;
    }
"""

# Navigation bar — Adobe-style top toolbar
NAV_BAR_STYLE = """
    QFrame#navBar {
        background-color: #1e1e1e;
        border-bottom: 1px solid #3a3a3a;
    }
    
    QLabel#appTitle {
        color: #2d8ceb;
        font-size: 13px;
        font-weight: bold;
        letter-spacing: 2px;
    }
"""

NAV_BUTTON_STYLE = """
    QPushButton {{
        background-color: transparent;
        color: {text_secondary};
        border: none;
        border-bottom: 2px solid transparent;
        padding: 0px 10px;
        font-size: 12px;
        font-weight: bold;
        letter-spacing: 1px;
    }}
    QPushButton:hover {{
        color: {text_primary};
        background-color: rgba(45, 140, 235, 0.08);
    }}
""".format(**COLORS)

NAV_BUTTON_ACTIVE_STYLE = """
    QPushButton {{
        background-color: rgba(45, 140, 235, 0.12);
        color: {accent};
        border: none;
        border-bottom: 2px solid {accent};
        padding: 0px 10px;
        font-size: 12px;
        font-weight: bold;
        letter-spacing: 1px;
    }}
""".format(**COLORS)

SIDEBAR_STYLE = """
/* ===== Sidebar Base ===== */
QFrame {
    background-color: #2d2d2d;
}

QLabel {
    color: #e0e0e0;
    font-size: 12px;
}

QLabel#sectionTitle {
    font-weight: bold;
    margin-top: 10px;
    margin-bottom: 4px;
}

/* ===== Button ===== */
QPushButton {
    background-color: #383838;
    color: #e0e0e0;
    border: 1px solid #4a4a4a;
    border-radius: 4px;
    padding: 8px;
    text-align: center;
}

QPushButton:hover {
    background-color: #404040;
    border-color: #555;
}

QPushButton:pressed {
    background-color: #505050;
}

QPushButton:disabled {
    background-color: #2a2a2a;
    color: #555;
    border-color: #333;
}

/* ===== ComboBox ===== */
QComboBox {
    background-color: #383838;
    color: #e0e0e0;
    border: 1px solid #4a4a4a;
    border-radius: 4px;
    padding: 6px;
}

QComboBox QAbstractItemView {
    background-color: #383838;
    color: #e0e0e0;
    selection-background-color: #2d8ceb;
}
"""