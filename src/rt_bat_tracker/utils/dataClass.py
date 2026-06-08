import threading


class SharedState:

    def __init__(self):

        self.lock = threading.Lock()

        self.sources = []
        self.latest_chunk = None
        self.detected = False
