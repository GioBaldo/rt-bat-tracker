from rt_bat_tracker.utils.point_class import Point
import numpy as np
import time
import logging

logger = logging.getLogger("EVENT")
logger.setLevel(logging.INFO)


class Event:

    def __init__(self, idx, start_time):
        self.points: list[Point] = []
        # IMPORTANT: time_adc must be used for sample alignment
        self.start_time_adc = start_time
        # IMPORTANT: time is used for timers. they are slightly different
        self.start_time = time.monotonic()  # (local time)
        self.event_name = f"Event {idx}"
        self.duration = None
        self.last_call_time = start_time
        self.sleep_timer = 0
        self.audio_file = []
        self.audio_fist_timestamp = None
        self.spectrogram = []

    def add_point(self, pos, timestamp, analytics):
        rel_ts = timestamp - self.start_time_adc
        idx = len(self.points)
        p = Point(pos, timestamp, rel_ts, idx=idx, analytics=analytics)
        p.score = (
            int(np.interp(abs(analytics.TDOA_sum), [1e-6, 1e-4], [127, 0]))
            if analytics.TDOA_sum is not None
            else 127
        )
        logger.info(f"POINT {p.idx} time: {p.rel_ts:.3f} s - score: {p.score}")
        self.points.append(p)

        return

    def update_sleep_timer(self):
        self.sleep_timer += time.monotonic - self.sleep_timer
        if self.sleep_timer > self._state.max_event_sleep:
            self.terminate_event()
        return

    def terminate_event(self):
        self.duration = time.monotonic() - self.start_time
