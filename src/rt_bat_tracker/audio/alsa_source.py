from re import match

import alsaaudio as alsa
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class AlsaAudioSource:
    """
    Captures live audio from a hardware device via Alsa in Raspian throug a focusrite Scarlett 18i20.
    Uses PyAlsaAudio library by @lassimmisch for bindings between alsa and python.

    PCM object needs to be configured with some values:
        PCM(type: int = PCM_PLAYBACK, mode: int = PCM_NORMAL, rate: int = 44100, channels: int = 2,
        format: int = PCM_FORMAT_S16_LE, periodsize: int = 32, periods: int = 4,
        device: str = 'default', cardindex: int = -1) -> PCM

    some of them can be change by the use and are given as inputs for this class, others are hard-coded in order
    to have the device properly working within the linux-alsa-focusrite fixed environment

    inputs:
        - queue
        - device
        - fs (rate)
        - channels
        - blocksize (periodsize)


    this audiosource can just be started with start() and stopped with stop(). blocks are continuously pushed to the shared
    queue audio_queue as an np.array of shape (blocksize, channels) as <np.float32> samples

    the hardcoded values are
    type: alsa.PCM_CAPTURE [1], the PCM is in CAPTURE mode for RECORDING (PLAYBACK otherwise)
    mode: alsa.PCM_NORMAL [0], the PCM is in NORMAL mode then BLOCKS the caller until a frame is full, nice to avoid empty reads
                (NONBLOCK otherwise)
    format: int = PCM_FORMAT_S32_LE actually the only sample format that works
    periods: int = 1, if more than one then a a larger number of blocks are passed each time
    cardindex: int = -1 NON USED
    """

    def __init__(self, state, device="hw:Gen,0", fs=192000, channels=8, blocksize=1024):
        self._state = state
        self.device = device  # resolve_input_device(device)
        self.fs = fs
        self.channels = channels
        self.blocksize = blocksize
        self.PCM = None
        self.chunknum = 1
        self.timestamp_type = "timeofday"  # "sync" | "raw" | "timeofday" | "monotonic"
        self.timestamp = None
        self.timer_bias = None
        self.timestamp_bias = None
        self.blocktime = self.blocksize / self.fs * 1000000000  # in ns

    def loop(self):
        """
        Block is red and data are pushed to the queue
        managed by te start function
        """
        if self.PCM.state() == 2:
            size, block = self.PCM.read()
            self.timestamp = self.get_timestamp()

            self.chunknum += 1

        elif self.PCM.state() == 3:

            size, block = self.PCM.read()

            if size < self.blocksize:
                self.chunknum += 1
                logger.error(
                    f"ALSA error: received block size {size} instead of {self.blocksize}"
                )
                return True
            else:
                xAr = self.to_nparray(block)
                drift = int(
                    (self.get_timestamp() - self.timestamp) * 100 / self.blocktime - 100
                )
                self.timestamp = self.get_timestamp()
                logger.debug(
                    f"[{self.chunknum}] ({self.PCM.state()}) ts[{int(self.timestamp/1000000)/1000}] x shape: {np.shape(xAr)} RMS: {self.compute_max(xAr)} drift = {drift}% cumulative drift: { int((self.counter_time() - self.timestamp) / 1000)/1000 } ms"
                )

                self._state.put_audio(
                    xAr, int(self.timestamp / 1000000) / 1000
                )  # timestamp is passed to the proc thread in s
                self.chunknum += 1
        else:
            logger.error(f"LOOP RUNNING BUT PCM STATE IS {self.PCM.state()}")
            return True

    def start(self):
        """
        called by the audio_input.run() selector, this funcion initializes the PCM object to stream
        audio from the Focusrite Scarlett 18i20 to the Raspberry Pi4 (raspian) through alsa.
        After the PCM is correctly initialized the loop() is activated for reading data
        """
        # try:
        self.PCM = alsa.PCM(
            type=alsa.PCM_CAPTURE,
            mode=alsa.PCM_NORMAL,
            rate=self.fs,
            channels=10,
            format=alsa.PCM_FORMAT_S32_LE,
            periodsize=self.blocksize,
            periods=10,
            device=self.device,
        )

        logger.info(self.PCM.info())
        self.timestamp_init()
        self._state.audio_stream_t_start = time.monotonic()

        try:
            while self.PCM is not None:
                if not self._state.stop_event.isSet():
                    if self.loop():
                        self.stop()
                        return __name__
                else:
                    self.stop()
                    logger.info(
                        "Alsa Audio source properly stopped after global stop event"
                    )
                    return None

        except KeyboardInterrupt:
            logger.info("PCM stopped by user")
            self.stop()
            return __name__

        except Exception as e:
            logger.error("Error in alsa_input: ", e)
            self.stop()
            return __name__

    # FUNCTIONS ======================================================

    def stop(self):
        """Closes the alsa stream and releases resources."""
        self.PCM.close()
        logger.info("PCM closed")

    def compute_max(self, x):
        mag = []
        thr = 0.01
        m = np.max(x.astype(np.float32), axis=0)
        for i in m:
            if i > thr:
                mag.append("A")
            else:
                mag.append("_")
        return mag

    def to_nparray(self, block):
        xAr = np.frombuffer(block, dtype=np.int32)
        xAr = xAr.reshape(-1, 10)[:, : self.channels]
        xAr = xAr.astype(np.float32) / (2**31)
        return xAr

    def get_timestamp(self):
        match self.timestamp_type:
            case "sync":
                return self.chunknum * self.blocktime
            case "raw":
                return (
                    self.PCM.htimestamp()[0] * 1000000000
                    + self.PCM.htimestamp()[1]
                    - self.timestamp_bias
                )
            case "timeofday":
                return (
                    self.PCM.htimestamp()[0] * 1000000000
                    + self.PCM.htimestamp()[1]
                    - self.timestamp_bias
                )
            case "monotonic":
                return (
                    self.PCM.htimestamp()[0] * 1000000000
                    + self.PCM.htimestamp()[1]
                    - self.timestamp_bias
                )

    def timestamp_init(self):
        match self.timestamp_type:
            case "sync":
                self.timestamp_bias = 0
                self.timer_bias = time.perf_counter_ns()

            case "raw":
                self.PCM.set_tstamp_mode(alsa.PCM_TSTAMP_ENABLE)
                self.PCM.set_tstamp_type(alsa.PCM_TSTAMP_TYPE_MONOTONIC_RAW)
                trash = self.PCM.read()
                self.timestamp_bias = (
                    self.PCM.htimestamp()[0] * 1000000000 + self.PCM.htimestamp()[1]
                )
                self.timer_bias = time.perf_counter_ns()

            case "timeofday":
                self.PCM.set_tstamp_mode(alsa.PCM_TSTAMP_ENABLE)
                self.PCM.set_tstamp_type(alsa.PCM_TSTAMP_TYPE_GETTIMEOFDAY)
                trash = self.PCM.read()
                self.timestamp_bias = (
                    self.PCM.htimestamp()[0] * 1000000000 + self.PCM.htimestamp()[1]
                )
                self.timer_bias = time.perf_counter_ns()

            case "monotonic":
                self.PCM.set_tstamp_mode(alsa.PCM_TSTAMP_ENABLE)
                self.PCM.set_tstamp_type(alsa.PCM_TSTAMP_TYPE_MONOTONIC)
                trash = self.PCM.read()
                self.timestamp_bias = (
                    self.PCM.htimestamp()[0] * 1000000000 + self.PCM.htimestamp()[1]
                )
                self.timer_bias = time.perf_counter_ns()

        self.timestamp = 0
        logger.debug(
            f"tstamp type: {self.PCM.get_tstamp_type()} [{self.timestamp_type}] set timestamp bias to {self.timestamp_bias} timer bias: {self.timer_bias} "
        )

    def counter_time(self):
        return time.perf_counter_ns() - self.timer_bias
