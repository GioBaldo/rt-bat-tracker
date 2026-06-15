import numpy as np


class Point:
    """probably this could become a simpler data collection like a list,
    at the moment I think it will only contain static informations of the point
    """

    def __init__(self, pos, abs_ts, rel_ts):

        self.pos = pos
        self.abs_ts = abs_ts  # trying to avoid using abs_ts. use rel_ts + event.start_time/start_time_adc instead for decoupling
        self.rel_ts = rel_ts
        self.color = None
        self.size = None
