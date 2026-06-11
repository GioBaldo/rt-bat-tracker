from rt_bat_tracker.utils.point_class import Point
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)


class Event:

    def __init__(self, state, idx, start_time=time.monotonic()):
        self.points: list[Point] = []
        self._state = state
        self.start_time = start_time
        self.event_name = str("Event ", idx)
        self.duration = None

        self.sleep_timer = 0

    def add_point(self, pos, timestamp):
        p = Point(pos, timestamp)
        self.points.append(p)
        return

    def get_points(self, time):
        active_points = []
        for p in self.points:
            age = time - p.abs_ts
            if age > 0:
                op = min(0, 1 - op / self._state.fade_time) if self._state.fade else 1
                p.color = (self._state.tail_color, op)
                active_points.append(p)

        return active_points

    def update_sleep_timer(self):
        self.sleep_timer += time.monotonic - self.sleep_timer
        if self.sleep_timer > self._state.max_event_sleep:
            self.terminate_event()
        return

    def terminate_event(self):
        self.duration = time.monotonic - self.start_time
        logger.info(
            "Event (%s) terminated - %d corrupted results reported",
            self.event_name,
            self._state.empty_res_count,
        )
        self
