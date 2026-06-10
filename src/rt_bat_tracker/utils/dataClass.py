import threading
import queue
import time
from dataclasses import dataclass, field
import logging
import math
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)


class SharedState:

    def __init__(self, cfg):

        self.audio_queue = queue.Queue(maxsize=cfg.audio_queue_maxsize)
        self.result_queue = queue.Queue(maxsize=cfg.results_queue_maxsize)

        self.gui_buffer = deque(maxlen=cfg.buffer_length)
        self.buffer_lock = threading.Lock()

        self.call_chunk = np.ndarray([])
        self.call_flag = False
        self.call_time = None

        self.stop_event = threading.Event()

        self.dropped_audio = 0
        self.dropped_results = 0

        self.t_start = None
        self.gui_t_start = None
        self.gui_running_flag = False

        logger.info("trying to load micxyz from: %s", cfg.micLayout_path)

        self.micxyz = np.loadtxt(cfg.micLayout_path, delimiter=",")

    def start(self):
        self.t_start = time.monotonic()
        print(f"timerStarted: {self.t_start}")

    def stop(self, caller="unspecified"):
        logger.warning("stop requested by %s", caller)
        self.stop_event.set()
        self.t_start = None

    def elapsed_time(self):
        if self.t_start is None:
            return 0.0
        return time.monotonic() - self.t_start

    # Audio queue functions
    def put_audio(self, audio_block, timestamp=None):
        try:
            self.audio_queue.put_nowait((audio_block, timestamp))
        except queue.Full:
            self.dropped_audio += 1
            logger.info(
                "Full audio queue, dropped audio blocks: %d", self.dropped_audio
            )

    def get_audio(self, timeout=0.1):
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            logger.debug("Empty audio queue, timeout after %.1f s", timeout)
            return None, None

    # Results queue functions
    def put_result(self, result, timestamp=None):
        try:
            self.result_queue.put_nowait((result, timestamp))
        except queue.Full:
            self.dropped_results += 1
            logger.debug(
                "Full results queue, dropped results: %d", self.dropped_results
            )

    def get_result(self, timeout=0.1):
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            logger.debug("Empty results queue, timeout after %.1f s", timeout)
            return None, None

    def write_buffer(self, result, timestamp):

        if self.stop_event.is_set():
            return
        with self.buffer_lock:
            if timestamp is None:
                # self.gui_buffer.append({"pos": None, "timestamp": None})
                return

            elif len(result) < 1:
                # self.gui_buffer.append({"pos": None, "timestamp": timestamp})
                return
            else:
                res = result[0]
                self.gui_buffer.append((np.array([res[0], res[1], res[2]]), timestamp))

        return True

    def read_buffer(self):
        with self.buffer_lock:
            all_points = np.array([p[0] for p in self.gui_buffer])
            all_times = np.array([p[1] for p in self.gui_buffer])
        return all_points, all_times
