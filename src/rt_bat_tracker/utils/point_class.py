import numpy as np


class Point:
    """probably this could become a simpler data collection like a list,
    at the moment I think it will only contain static informations of the point
    """

    def __init__(self, pos, abs_ts):

        self.pos = pos
        self.abs_ts = abs_ts
        self.color = None
        self.size = None
