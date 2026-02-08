SIDEBAR_STYLE = """
/* ===== Sidebar Base ===== */
QFrame {
    background-color: #2b2b2b;
}

QLabel {
    color: #e0e0e0;
    font-size: 13px;
}

QLabel#sectionTitle {
    font-weight: bold;
    margin-top: 10px;
    margin-bottom: 4px;
}

/* ===== Button ===== */
QPushButton {
    background-color: #1e1e1e;
    color: #e0e0e0;
    border: 1px solid #444;
    border-radius: 6px;
    padding: 8px;
    text-align: center;
}

QPushButton:hover {
    background-color: #353535;
}

QPushButton:pressed {
    background-color: #555;
}

QPushButton:disabled {
    background-color: #2a2a2a;
    color: #777;
    border-color: #333;
}

/* ===== ComboBox ===== */
QComboBox {
    background-color: #1e1e1e;
    color: #e0e0e0;
    border: 1px solid #444;
    border-radius: 6px;
    padding: 6px;
}

QComboBox QAbstractItemView {
    background-color: #2a2a2a;
    color: #e0e0e0;
    selection-background-color: #555;
}
"""