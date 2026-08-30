import threading
import queue
import time
from dataclasses import dataclass, field
import logging
import math
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SharedState:

    def __init__(self, cfg):
        self.is_live = True

        self.audio_queue = queue.Queue(maxsize=cfg.audio_queue_maxsize)
        self.result_queue = queue.Queue(maxsize=cfg.results_queue_maxsize)

        self.threshold = cfg.threshold

        self.event_wav_buffer = deque(maxlen=cfg.event_wav_buffer_maxsize)
        self.buffer_lock = threading.Lock()

        self.call_chunk = np.ndarray([])
        self.call_flag = False
        self.call_time = None

        self.stop_event = threading.Event()

        self.dropped_audio = 0
        self.dropped_results = 0

        self.max_rms = 0
        self.avg_rms = 0
        self.EMA_rms = 0

        self.t_start = None
        self.audio_stream_t_start = None
        self.gui_t_start = None
        self.gui_running_flag = False
        self.fade = True
        self.fade_time = 1
        self.tail_color = np.array([1.0, 1.0, 0.0])

        self.SAVE_RESULTS = cfg.SAVE_RESULTS

        self.ROTATION_X = 80
        self.ROTATION_Z = 20
        self.H_DISPLACEMENT = 1

        logger.info("trying to load micxyz from: %s", cfg.micLayout_path)
        root_xyz = np.loadtxt(cfg.micLayout_path, delimiter=",")
        self.micxyz = self.rotate_coords(root_xyz, self.ROTATION_X, self.ROTATION_Z, self.H_DISPLACEMENT)

        # stats
        self.empty_res_count = 0

    def start(self):
        self.t_start = time.monotonic()

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
        if self.stop_event.is_set():
            return None, None
        try:
            result, timestamp = self.result_queue.get(timeout=timeout)

            if timestamp is None:
                return None, None

            elif len(result) < 1:
                self.empty_res_count += 1
                return None, None
            else:
                res = result[0]
                return np.array([res[0], res[1], res[2]]), timestamp

        except queue.Empty:
            logger.debug("Empty results queue, timeout after %.1f s", timeout)
            return None, None

    def write_wav_buffer(self, block):
        """
        called from processing, writes blocks in a deque buffer.
        if data are not grabbed from session older ones are dropped

        Args:
            block (np.array): shape [blocksize, channels]
        """
        logger.debug(f"writing wav_buffer with {np.shape(block)} samples")
        with self.buffer_lock:
            self.event_wav_buffer.append((block))

    def grab_wav_buffer(self):
        """
        returns frames stored in the buffer since the last grab
        as <list> of elements [blocksize, channels]
        """
        with self.buffer_lock:
            items = list(self.event_wav_buffer)
            self.event_wav_buffer.clear()
        return items

    def rotate_coords(self, root_xyz, rotation_x_deg, rotation_z_deg, h_displacement):
        """
        Rotates the microphone coordinates around the X and Z axes and translates them along the Z axis.
        """
        rad_x = np.radians(rotation_x_deg)
        rad_z = np.radians(rotation_z_deg)

        Rx = np.array([
            [1,           0,            0],
            [0, np.cos(rad_x), -np.sin(rad_x)],
            [0, np.sin(rad_x),  np.cos(rad_x)]
        ])

        Rz = np.array([
            [np.cos(rad_z), -np.sin(rad_z), 0],
            [np.sin(rad_z),  np.cos(rad_z), 0],
            [0,              0,             1]
        ])

        # Combine the rotation matrices
        R_total = Rz @ Rx

        #Matrix multiplication to rotate the coordinates
        rotated_xyz = root_xyz @ R_total.T

        # translation along the Z axis (height displacement)
        rotated_xyz[:, 2] += h_displacement

        return rotated_xyz