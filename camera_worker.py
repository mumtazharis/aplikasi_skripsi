import cv2
from PySide6.QtCore import QThread, Signal

class CameraWorker(QThread):
    frame_ready = Signal(object)

    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)

        while self.running:
            ret, frame = cap.read()
            if ret:
                self.frame_ready.emit(frame)

        cap.release()

    def stop(self):
        self.running = False
        self.wait()
