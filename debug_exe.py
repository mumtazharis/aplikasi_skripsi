
import sys
import os
import traceback

meipass = r"d:\Belajar\aplikasi_skripsi\dist\Expression Analyzer\_internal"
sys._MEIPASS = meipass
sys.path.insert(0, meipass)
os.chdir(r"d:\Belajar\aplikasi_skripsi\dist\Expression Analyzer")

# Mock for resource_path, since resource_path checks sys._MEIPASS
from ml.pipeline import init_models

try:
    init_models(device="cpu")
    print("SUCCESS")
except Exception as e:
    print(f"FAILED: {e}")
    traceback.print_exc()

