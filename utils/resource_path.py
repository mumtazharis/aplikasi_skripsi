import sys
import os

def resource_path(relative_path):
    """ Dapatkan absolute path ke resource, bekerja untuk dev dan PyInstaller """
    try:
        # PyInstaller membuat folder temp di _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)