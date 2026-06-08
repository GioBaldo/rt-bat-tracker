# file: examples/audio_block_thread.py
import threading
import time
from rt_bat_tracker.audio.sources import AudioFileSource
import keyboard


class AudioBlockMonitor:
    def __init__(self, source):
        self.start_time = None
        self.source = source
        self.running = False
        self.block_time = None
        self.thread = threading.Thread(target=self._run)

    def start(self):
        self.block_time = self.source.get_blocksize() / self.source.fs * 1000  # in ms
        print(
            "Audio block monitor started. Press SPACE to stop. Block time: {:.2f} ms".format(
                self.block_time
            )
        )
        self.running = True
        self.start_time = time.time() * 1000  # convert to ms
        self.thread.start()

        # Listen for spacebar
        keyboard.add_hotkey("space", self.stop)

    def stop(self):
        self.running = False
        print("Stopping monitor...")

    def _run(self):
        while self.running:
            chunk, chunknum = self.source.get_block()

            if chunk is not None:
                track_time = chunknum * self.block_time
                time_now = time.time() * 1000 - self.start_time
                print(
                    f"N = {chunknum}: [{chunk.shape}] - time = {track_time:.2f} ms / {time_now:.2f} ms - delay = {time_now - track_time:.2f} ms"
                )
                time.sleep(
                    max(
                        0,
                        (self.block_time * (chunknum + 1) - time_now) / 1100,
                    ),
                )  # to admit threading and not hog the CPU
            else:
                time.sleep(self.block_time / 1000)  # prevent busy waiting


if __name__ == "__main__":
    # Example usage with an audio file source
    print("run from main.")
