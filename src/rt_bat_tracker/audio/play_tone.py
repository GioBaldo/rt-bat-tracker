import alsaaudio as alsa
import numpy as np
import time
import threading
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class PlayTone:
    def __init__(self, state, cfg):
        self.cfg = cfg
        self.state = state
        self.channels = [0, 1]
        self.blocksize = 4096
        self.PCM = alsa.PCM(
            type=alsa.PCM_PLAYBACK,
            mode=alsa.PCM_NORMAL,
            rate=self.cfg.fs,
            format=alsa.PCM_FORMAT_S32_LE,
            periodsize=self.blocksize,
            periods=100,
            device="hw:Gen,0",
        )

    def play(self, frequency=30000, duration=5, rate=1):
        """Play a tone of given frequency and duration."""
        samples = int((self.cfg.fs * duration / 1000) / self.blocksize) * self.blocksize
        tone = self.make_tone(frequency, samples)
        print(self.PCM.info())
        print(
            f"requested tone frequency: {frequency}Hz, duration: {duration}ms, rate: {rate} calls/sec: actual tone length: {len(tone)} = {len(tone)/self.blocksize} x {self.blocksize} "
        )
        interval = min(1 / rate - (duration / 1000), 0.005)
        beep_thread = threading.Thread(target=self.beep, args=(tone, interval))
        beep_thread.start()

    def make_tone(self, frequency, samples):
        """Generate a sine wave tone."""
        t = np.linspace(0, samples / self.cfg.fs, samples, endpoint=False)
        tone = 10000 * np.sin(2 * np.pi * frequency * t).astype(np.int32)

        return tone.tobytes()

    def beep(self, tone, interval):

        try:
            num_chunks = len(tone) // self.blocksize

            if num_chunks == 0:
                print(" tone too short for blocksize")
                return

            while not self.state.stop_event.isSet():

                for chunk in range(num_chunks):

                    block = tone[chunk * self.blocksize : (chunk + 1) * self.blocksize]

                    if len(block) != self.blocksize:
                        continue  # safety guard

                    frame = np.zeros(self.blocksize * 10, dtype=np.int32)

                    for ch in self.channels:
                        frame[ch::10] = block

                    self.PCM.write(frame.tobytes())

                time.sleep(interval)

        except Exception as e:
            print("beep thread crashed:", repr(e))
