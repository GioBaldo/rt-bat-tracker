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
        self.channels = [0]
        self.blocksize = 4096
        self.chunk_time = np.around(self.blocksize * 1000 / self.cfg.fs, 2)
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
        samples = (
            int(np.ceil((self.cfg.fs * duration / 1000) / self.blocksize))
            * self.blocksize
        )
        # tone = self.make_tone(frequency, samples)
        tone = self.make_sweep(frequency, 2000, samples)
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
        tone = (1e9 * np.sin(2 * np.pi * frequency * t)).astype(np.int32)
        n = len(tone)
        ramp = n // 2
        window = np.ones(n)
        window[:ramp] = np.linspace(0, 1, ramp)
        window[-ramp:] = np.linspace(1, 0, ramp)
        tone = (tone * window).astype(np.int32)
        print(tone[:10])
        return tone

    def make_sweep(self, start_freq, end_freq, samples):
        """Generate a linear sine sweep."""

        t = np.linspace(0, samples / self.cfg.fs, samples, endpoint=False)
        duration = samples / self.cfg.fs

        # Linear chirp phase
        k = (end_freq - start_freq) / duration
        phase = 2 * np.pi * (start_freq * t + 0.5 * k * t**2)

        sweep = (1e9 * np.sin(phase)).astype(np.int32)

        # Apply the same trapezoidal envelope
        n = len(sweep)
        ramp = n // 2
        window = np.ones(n)
        window[:ramp] = np.linspace(0, 1, ramp)
        window[-ramp:] = np.linspace(1, 0, ramp)

        sweep = (sweep * window).astype(np.int32)

        return sweep

    def beep(self, tone, interval):

        try:
            num_chunks = len(tone) // self.blocksize

            if num_chunks == 0:
                logger.warning(" tone too short for blocksize")
                return

            while not self.state.stop_event.isSet():
                logger.info(
                    f"BEEP interval: {interval}, num_chunks: {num_chunks}, duration: {np.around(num_chunks*self.chunk_time, 2)}ms"
                )
                # time.sleep(1)
                for ch in self.channels:
                    logger.debug(f"play ch {ch} .. interval {interval}")
                    self.output(tone, ch, num_chunks, 1)
                    print("sleeping interval")
                    time.sleep(interval)

        except Exception as e:
            logger.error("beep thread crashed:", repr(e))

    def output(self, tone, ch, num_chunks, amp):

        for chunk in range(num_chunks):

            block = tone[chunk * self.blocksize : (chunk + 1) * self.blocksize]

            if len(block) != self.blocksize:
                continue  # safety guard

            frame = np.zeros(self.blocksize * 10, dtype=np.int32)

            frame[ch::10] = block * amp

            self.PCM.write(frame.tobytes())
            logger.debug(
                f"write on ch {ch} chunk {chunk} .. time {time.monotonic()} frame: {frame[100:106]}"
            )
