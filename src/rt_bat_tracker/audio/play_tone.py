import alsaaudio as alsa
import numpy as np
import time
import threading
import logging

logger = logging.getLogger("PLAYER")
logger.setLevel(logging.INFO)


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

    def play(self, duration=5, frequency=30000, rate=1):
        """Play a tone of given frequency and duration."""
        samples = int((self.cfg.fs * duration / 1000) / self.blocksize) * self.blocksize
        tone = self.make_tone(frequency, samples)
        logger.info(self.PCM.info())
        logger.info(
            f"requested tone frequency: {frequency}Hz, duration: {duration}ms, rate: {rate} calls/sec: actual tone length: {len(tone)} = {len(tone)/self.blocksize} x {self.blocksize} "
        )
        # interval = min(1 / rate - (duration / 1000), 0.005)
        interval = 1 / rate
        self.beep(tone, interval)

    def make_tone(self, frequency, samples):
        """Generate a sine wave tone."""
        t = np.linspace(0, samples / self.cfg.fs, samples, endpoint=False)
        tone = (100000000000 * np.sin(2 * np.pi * frequency * t)).astype(np.int32)

        return tone

    def beep(self, tone, interval):

        try:
            num_chunks = len(tone) // self.blocksize

            if num_chunks == 0:
                logger.warning(" tone too short for blocksize")
                return

            while not self.state.stop_event.isSet():
                logger.info(f"BEEP interval: {interval}")

                for chunk in range(num_chunks):

                    block = tone[chunk * self.blocksize : (chunk + 1) * self.blocksize]

                    if len(block) != self.blocksize:
                        continue  # safety guard

                    frame = np.zeros(self.blocksize * 10, dtype=np.int32)

                    for ch in self.channels:
                        frame[ch::10] = block

                    logger.debug(
                        f"frame shape: {frame.shape}, block shape: {block.shape}, max block: {block[100:120]} frame: {frame[1000:1020]}"
                    )
                    self.PCM.write(frame.tobytes())

                time.sleep(interval)

        except Exception as e:
            logger.error("beep thread crashed:", repr(e))
