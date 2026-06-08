import threading
import keyboard
import time


class GUIUpdate:
    def __init__(self, update_interval=0.1):
        self.update_interval = update_interval
        self.running = False
        self.thread = threading.Thread(target=self._run)

    def start(self):
        self.running = True
        self.thread.start()
        print("GUI update thread started.")
        keyboard.add_hotkey("space", self.stop)

    def stop(self):
        self.running = False
        print("Stopping GUI update thread...")

    def _run(self):
        while self.running:
            # Here you would put the code to update your GUI elements
            # For example, refreshing plots, updating labels, etc.
            print("Updating GUI...")
            time.sleep(self.update_interval)
