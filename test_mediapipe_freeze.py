
import sys
import os

if hasattr(sys, "_MEIPASS"):
    print("Mocking MEIPASS")
else:
    sys._MEIPASS = r"d:\Belajar\aplikasi_skripsi\dist\Expression Analyzer\_internal"
    sys.frozen = True

import mediapipe as mp
print(mp.solutions.face_mesh.FaceMesh())

