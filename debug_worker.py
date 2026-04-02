
import sys
import os
import traceback

meipass = r"d:\Belajar\aplikasi_skripsi\dist\Expression Analyzer\_internal"
sys._MEIPASS = meipass
sys.path.insert(0, meipass)
os.chdir(r"d:\Belajar\aplikasi_skripsi\dist\Expression Analyzer")

from workers.prediction_worker import PredictionWorker

try:
    # Use an actual video from recording or a short mock video
    worker = PredictionWorker(r"d:\Belajar\aplikasi_skripsi\test.mp4")
    # worker.run() will execute the whole pipeline
except Exception as e:
    print(traceback.format_exc())

