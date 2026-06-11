from rt_bat_tracker.utils.event_class import Event
import time


class Session:

    def __init__(self, cfg, state, name=time.asctime()):
        self._state = state
        self._cfg = cfg
        self.listEvents: list[Event] = []
        self.active_event = None
        self.session_name = name

    def new_Event(self, time):
        self.active_event = Event(self._state, time, len(self.listEvents) + 1)
        self.listEvents.append()

        self._state.empty_res_count = 0
        return self.active_event
