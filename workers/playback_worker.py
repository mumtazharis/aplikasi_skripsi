import os
import csv
import cv2
import time
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from PySide6.QtCore import QThread, Signal, QMutex
import numpy as np


# ================================================================
# Deteksi jumlah core CPU secara otomatis
# os.cpu_count() = logical threads (misal 12 untuk 6-core HT)
# Gunakan setengah untuk physical core estimation
# ================================================================
_LOGICAL_CORES = os.cpu_count() or 4
_PHYSICAL_CORES = max(2, _LOGICAL_CORES // 2)
_IO_WORKERS = _PHYSICAL_CORES          # untuk parallel I/O (FrameCache)
_CV_THREADS = _PHYSICAL_CORES          # untuk OpenCV internal ops

cv2.setNumThreads(_CV_THREADS)


class FrameCache:
    """
    Cache LRU untuk frame gambar dari folder.
    OPTIMIZED: ThreadPoolExecutor untuk parallel I/O.
    cv2.imread() melepas GIL → benar-benar paralel di C++ layer.
    """

    def __init__(self, source_path, frame_files, cache_size=200, num_workers=_IO_WORKERS):
        self.source_path = source_path
        self.frame_files = frame_files
        self.cache_size = cache_size
        self.cache = OrderedDict()  # idx -> np.ndarray (RGB)
        self.lock = threading.Lock()

        # ThreadPoolExecutor untuk parallel frame loading
        self.num_workers = num_workers
        self.executor = ThreadPoolExecutor(
            max_workers=num_workers,
            thread_name_prefix="frame_io"
        )
        self._pending = set()  # set of idx yang sedang di-load
        self._shutdown = False

    def get(self, idx):
        """Ambil frame dari cache. Jika belum ada, baca langsung (fallback)."""
        with self.lock:
            if idx in self.cache:
                self.cache.move_to_end(idx)
                return self.cache[idx]

        # Cache miss — baca langsung (blocking, tapi cepat karena NVMe)
        frame = self._read_frame(idx)
        if frame is not None:
            with self.lock:
                self.cache[idx] = frame
                self._evict()
        return frame

    def preload_range(self, start_idx, count=100):
        """
        Submit batch frame ke thread pool untuk parallel loading.
        Setiap frame di-submit sebagai task independen.
        """
        if self._shutdown:
            return

        total = len(self.frame_files)
        for i in range(count):
            idx = start_idx + i
            if idx >= total:
                break

            # Skip yang sudah ada di cache atau sedang di-load
            with self.lock:
                if idx in self.cache:
                    continue
            if idx in self._pending:
                continue

            self._pending.add(idx)
            self.executor.submit(self._load_and_cache, idx)

    def _load_and_cache(self, idx):
        """Worker task: baca frame dari disk dan masukkan ke cache."""
        try:
            if self._shutdown:
                return
            frame = self._read_frame(idx)
            if frame is not None and not self._shutdown:
                with self.lock:
                    self.cache[idx] = frame
                    self._evict()
        finally:
            self._pending.discard(idx)

    def _read_frame(self, idx):
        """Baca satu frame dari disk. cv2.imread melepas GIL → true parallelism."""
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
        # Tidak bisa cancel futures yang sudah submitted, tapi kita bisa
        # bersihkan cache dan set shutdown flag sementara
        with self.lock:
            self.cache.clear()
        self._pending.clear()

    def stop(self):
        """Shutdown thread pool. Dipanggil saat playback selesai."""
        self._shutdown = True
        self.executor.shutdown(wait=False, cancel_futures=True)


class PlaybackWorker(QThread):
    """
    Worker thread untuk memutar ulang video dan sinkronisasi dengan data CSV.
    Mendukung play/pause, seek, dan speed control.
    Menggunakan FrameCache (multi-threaded) untuk folder source.
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

        # Pre-scaling target (width, height) — set dari UI thread
        self._display_size = None  # tuple (w, h) or None

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

            # Buat cache multi-threaded (auto-detect workers + cache 200 frame)
            self.frame_cache = FrameCache(
                source_path, self.frame_files,
                cache_size=200, num_workers=_IO_WORKERS
            )
            # Preload awal — 100 frame secara paralel
            self.frame_cache.preload_range(0, 100)
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
                    self.frame_cache.preload_range(seek_target, 100)
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

                # Trigger preload ahead (setiap 30 frame, 100 frame ke depan)
                if self.is_folder and self.frame_cache:
                    if self.current_frame_idx - last_preload_idx >= 30:
                        self.frame_cache.preload_range(self.current_frame_idx, 100)
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

    def set_display_size(self, width, height):
        """Set target display size untuk pre-scaling di worker thread."""
        if width > 0 and height > 0:
            self._display_size = (width, height)
        else:
            self._display_size = None

    def _read_frame_at(self, idx):
        """Baca frame pada index tertentu, pre-scale jika display_size di-set."""
        frame = None
        if self.is_folder:
            if self.frame_cache:
                frame = self.frame_cache.get(idx)
        else:
            if self.cap and self.cap.isOpened():
                current_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                if current_pos != idx:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame_bgr = self.cap.read()
                if ret:
                    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Pre-scale di worker thread (off main thread)
        if frame is not None and self._display_size is not None:
            dw, dh = self._display_size
            fh, fw = frame.shape[:2]
            if fw != dw or fh != dh:
                # Keep aspect ratio
                scale = min(dw / fw, dh / fh)
                new_w = int(fw * scale)
                new_h = int(fh * scale)
                if new_w > 0 and new_h > 0:
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        return frame


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
