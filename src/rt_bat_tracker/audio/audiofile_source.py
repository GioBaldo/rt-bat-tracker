import soundfile as sf
import numpy as np
import logging
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AudioFileSource:
    """
    Reads audio from a file and feeds it into the shared queue
    at the correct real-time pace, simulating a live hardware source.

    Timing is based on the file's sample rate: each block is delivered
    at the exact wall-clock interval it would arrive from a real ADC.
    A drift-correction mechanism keeps timing accurate over long files
    by anchoring each block to an absolute t_start reference instead of
    using a fixed sleep — which would accumulate error over time.

    The synthetic timestamp passed to put_audio() mirrors the role of
    time_info.inputBufferAdcTime in RealtimeAudioSource, so the processing
    thread can use identical TDOA logic regardless of the source.
    """

    def __init__(self, state, file_path, block_size=2048, loaded_durn=None):
        self._state = state
        self.device = file_path
        self.blocksize = block_size
        self._chunknum = 0

        # Load audio file (optionally truncated to loaded_durn seconds)
        info = sf.info(file_path)
        self.fs = info.samplerate
        stop_sample = int(self.fs * loaded_durn) if loaded_durn else None
        self.audio = sf.read(
            file_path, stop=stop_sample, dtype="float32", always_2d=True
        )[0]
        # always_2d=True guarantees shape (samples, channels) even for mono files

        # Duration of one block in seconds — the pacing interval
        self._block_duration = self.blocksize / self.fs
        self.channels = self.audio.shape[1]

        logger.info(
            "AudioFileSource ready — file=%s fs=%d samples=%d channels=%d",
            file_path,
            self.fs,
            self.audio.shape[0],
            self.channels,
        )

    def start(self):
        """
        Iterates over the file block by block, pushing each block into
        the shared queue at the correct real-time pace.

        Blocks the calling thread until the file is exhausted or
        state.stop_event is set — same blocking behaviour as
        RealtimeAudioSource.start().
        """
        total_samples = self.audio.shape[0]

        # Absolute reference time for drift correction.
        # Every block's sleep is computed relative to this t_start,
        # so timing errors don't accumulate across blocks.
        t_start = time.monotonic()

        logger.info(
            "AudioFileSource started — block_duration=%.3f ms",
            self._block_duration * 1000,
        )

        for idx, start in enumerate(
            range(0, total_samples - self.blocksize, self.blocksize)
        ):

            if self._state.stop_event.is_set():
                break

            b = self.audio[start : start + self.blocksize, :]
            block = np.array(b, dtype=np.float32)

            # Synthetic ADC timestamp: position in file expressed in seconds.
            # Matches the role of time_info.inputBufferAdcTime in the realtime source.
            synthetic_timestamp = start / self.fs

            self._state.put_audio(block.copy(), synthetic_timestamp)
            self._chunknum += 1

            # logger.debug(
            #     "Delivered block %d (samples %d-%d) timestamp %.3f s",
            #     idx,
            #     start,
            #     start + self.block_size,
            #     synthetic_timestamp,
            # )

            # Drift-corrected sleep:
            # compute when this block *should* have been delivered and sleep
            # only the remaining delta — self-correcting on every iteration.
            next_block_time = t_start + (idx + 1) * self._block_duration
            sleep_time = next_block_time - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)

        logger.warning(
            "AudioFileSource finished — chunks delivered: %d  dropped: %d",
            self._chunknum,
            self._state.dropped_audio,
        )

        # File exhausted: signal shutdown to all other threads
        self._state.stop(__name__)

    def stop(self):
        """
        No-op: stop is handled via state.stop_event checked in the start() loop.
        Exists to keep the interface symmetric with RealtimeAudioSource.
        """
        pass
