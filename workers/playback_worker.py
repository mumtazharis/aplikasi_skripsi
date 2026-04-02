import os
import csv
import cv2
import time
import threading
from collections import OrderedDict
from PySide6.QtCore import QThread, Signal, QMutex
import numpy as np


class FrameCache:
    """
    Cache LRU untuk frame gambar dari folder.
    Preload frame-frame berikutnya di background thread agar playback mulus.
    """

    def __init__(self, source_path, frame_files, cache_size=60):
        self.source_path = source_path
        self.frame_files = frame_files
        self.cache_size = cache_size
        self.cache = OrderedDict()  # idx -> np.ndarray (RGB)
        self.lock = threading.Lock()
        self.preload_thread = None
        self.preload_running = False

    def get(self, idx):
        """Ambil frame dari cache. Jika belum ada, baca langsung (fallback)."""
        with self.lock:
            if idx in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(idx)
                return self.cache[idx]

        # Cache miss — baca langsung
        frame = self._read_frame(idx)
        if frame is not None:
            with self.lock:
                self.cache[idx] = frame
                self._evict()
        return frame

    def preload_range(self, start_idx, count=40):
        """Preload frame dari start_idx hingga start_idx + count di background."""
        # Stop existing preload
        self.preload_running = False
        if self.preload_thread and self.preload_thread.is_alive():
            self.preload_thread.join(timeout=0.5)

        self.preload_running = True
        self.preload_thread = threading.Thread(
            target=self._preload_worker,
            args=(start_idx, count),
            daemon=True,
        )
        self.preload_thread.start()

    def _preload_worker(self, start_idx, count):
        """Background worker yang membaca frame ke cache."""
        total = len(self.frame_files)
        for i in range(count):
            if not self.preload_running:
                break

            idx = start_idx + i
            if idx >= total:
                break

            # Skip jika sudah ada di cache
            with self.lock:
                if idx in self.cache:
                    continue

            frame = self._read_frame(idx)
            if frame is not None:
                with self.lock:
                    self.cache[idx] = frame
                    self._evict()

    def _read_frame(self, idx):
        """Baca satu frame dari disk."""
        if 0 <= idx < len(self.frame_files):
            filepath = os.path.join(self.source_path, self.frame_files[idx])
            frame_bgr = cv2.imread(filepath)
            if frame_bgr is not None:
                return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return None

    def _evict(self):
        """Hapus frame terlama jika cache penuh."""
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)

    def invalidate(self):
        """Bersihkan cache (misal saat seek jauh)."""
        self.preload_running = False
        if self.preload_thread and self.preload_thread.is_alive():
            self.preload_thread.join(timeout=0.5)
        with self.lock:
            self.cache.clear()

    def stop(self):
        self.preload_running = False
        if self.preload_thread and self.preload_thread.is_alive():
            self.preload_thread.join(timeout=1.0)


class PlaybackWorker(QThread):
    """
    Worker thread untuk memutar ulang video dan sinkronisasi dengan data CSV.
    Mendukung play/pause, seek, dan speed control.
    Menggunakan FrameCache untuk folder source agar playback mulus.
    """
    frame_ready = Signal(np.ndarray, int)   # (frame_rgb, frame_index)
    playback_finished = Signal()
    error = Signal(str)

    def __init__(self):
        super().__init__()
        self.running = False
        self.playing = False
        self.mutex = QMutex()

        # Source
        self.source_path = None
        self.is_folder = False
        self.fps = 30.0
        self.speed = 1.0

        # State
        self.current_frame_idx = 0
        self.total_frames = 0
        self.seek_target = -1  # -1 = no seek request

        # Folder source
        self.frame_files = []
        self.frame_cache = None

        # Video source
        self.cap = None

    def load_source(self, source_path, is_folder, fps=30.0):
        """Load video atau folder frame."""
        self.source_path = source_path
        self.is_folder = is_folder
        self.fps = fps if fps > 0 else 30.0
        self.current_frame_idx = 0

        if is_folder:
            valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
            raw_files = [f for f in os.listdir(source_path) if os.path.splitext(f)[1].lower() in valid_ext]
            self.frame_files = sorted(raw_files, key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
            self.total_frames = len(self.frame_files)

            # Buat cache untuk folder source
            self.frame_cache = FrameCache(source_path, self.frame_files, cache_size=80)
            # Preload awal
            self.frame_cache.preload_range(0, 60)
        else:
            if self.cap:
                self.cap.release()
            self.cap = cv2.VideoCapture(source_path)
            if not self.cap.isOpened():
                self.error.emit(f"Gagal membuka video: {source_path}")
                return False
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            if actual_fps > 0:
                self.fps = actual_fps

        return self.total_frames > 0

    def set_speed(self, speed):
        self.mutex.lock()
        self.speed = max(0.25, min(4.0, speed))
        self.mutex.unlock()

    def seek(self, frame_idx):
        self.mutex.lock()
        self.seek_target = max(0, min(frame_idx, self.total_frames - 1))
        self.mutex.unlock()

    def play(self):
        self.mutex.lock()
        self.playing = True
        self.mutex.unlock()

    def pause(self):
        self.mutex.lock()
        self.playing = False
        self.mutex.unlock()

    def toggle_play(self):
        self.mutex.lock()
        self.playing = not self.playing
        self.mutex.unlock()

    def stop(self):
        self.running = False
        self.playing = False
        if self.frame_cache:
            self.frame_cache.stop()
        self.wait()

    def run(self):
        self.running = True
        last_preload_idx = -1

        while self.running:
            self.mutex.lock()
            is_playing = self.playing
            speed = self.speed
            seek_target = self.seek_target
            self.seek_target = -1  # consume seek request
            self.mutex.unlock()

            # Handle seek
            if seek_target >= 0:
                self.current_frame_idx = seek_target
                if not self.is_folder and self.cap:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, seek_target)

                # Untuk folder: preload di sekitar posisi seek
                if self.is_folder and self.frame_cache:
                    self.frame_cache.preload_range(seek_target, 60)
                    last_preload_idx = seek_target

                frame = self._read_frame_at(self.current_frame_idx)
                if frame is not None:
                    self.frame_ready.emit(frame, self.current_frame_idx)

            # Handle playback
            if is_playing:
                if self.current_frame_idx >= self.total_frames:
                    self.mutex.lock()
                    self.playing = False
                    self.mutex.unlock()
                    self.playback_finished.emit()
                    self.msleep(50)
                    continue

                # Timing: catat waktu sebelum baca frame
                t_start = time.perf_counter()

                frame = self._read_frame_at(self.current_frame_idx)
                if frame is not None:
                    self.frame_ready.emit(frame, self.current_frame_idx)

                self.current_frame_idx += 1

                # Trigger preload ahead (setiap 20 frame)
                if self.is_folder and self.frame_cache:
                    if self.current_frame_idx - last_preload_idx >= 20:
                        self.frame_cache.preload_range(self.current_frame_idx, 60)
                        last_preload_idx = self.current_frame_idx

                # Frame timing: kompensasi waktu baca
                target_delay_ms = (1000.0 / self.fps) / speed
                elapsed_ms = (time.perf_counter() - t_start) * 1000
                remaining_ms = target_delay_ms - elapsed_ms

                if remaining_ms > 1:
                    self.msleep(int(remaining_ms))
            else:
                # Tidak playing, sleep untuk hemat CPU
                self.msleep(50)

        # Cleanup
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.frame_cache:
            self.frame_cache.stop()
            self.frame_cache = None

    def _read_frame_at(self, idx):
        """Baca frame pada index tertentu."""
        if self.is_folder:
            # Gunakan cache untuk folder source
            if self.frame_cache:
                return self.frame_cache.get(idx)
        else:
            if self.cap and self.cap.isOpened():
                # Ensure position is correct
                current_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                if current_pos != idx:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

                ret, frame_bgr = self.cap.read()
                if ret:
                    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return None


def load_csv_data(csv_path):
    """
    Helper function untuk load data CSV prediksi.
    Returns: list of dicts, satu per frame.
    """
    data = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = dict(row)
            for k, v in parsed.items():
                if k in ["frame_index", "face_x1", "face_y1", "face_x2", "face_y2"]:
                    parsed[k] = int(v) if v else 0
                elif k in ["timestamp_ms", "macro_confidence", "micro_confidence"]:
                    parsed[k] = float(v) if v else 0.0
                elif k in ["macro_label", "micro_label"]:
                    parsed[k] = v if v else "N/A"
                elif k.startswith("area_"):
                    parsed[k] = float(v) if v else 0.0
            data.append(parsed)
    return data


def load_meta(csv_path):
    """
    Load metadata file yang bersebelahan dengan CSV.
    Returns: dict with keys like 'source', 'is_folder', 'fps', etc.
    """
    meta_path = csv_path.replace(".csv", "_meta.txt")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if "=" in line:
                    key, value = line.split("=", 1)
                    meta[key.strip()] = value.strip()
    return meta
