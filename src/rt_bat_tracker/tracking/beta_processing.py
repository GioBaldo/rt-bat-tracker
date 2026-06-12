"""
processing.py

Reads audio blocks from SharedState.audio_queue and runs
signal evaluation on each block.

The processing loop uses a blocking get() with timeout so the thread
yields CPU to other threads while waiting for new data — no busy-waiting.
Results are written to SharedState.result_queue via state.put_result().

Entry point for the processing thread: run(state, cfg)
"""

import logging
import queue

import numpy as np
from scipy import signal

from rt_bat_tracker.tracking.localisation_mpr2003 import tristar_mellen_pachter
from rt_bat_tracker.tracking.common_functions import calc_rms, calc_multich_delays

# import librosa
# from scipy import signal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Processor class
# ---------------------------------------------------------------------------


class AudioProcessor:
    """
    Consumes audio blocks from the shared queue and runs
    signal evaluations on each block.

    Designed to run in a single dedicated thread.
    All methods are called sequentially — no internal threading.
    """

    def __init__(self, state, cfg):
        self._state = state
        self.fs = cfg.fs
        self.channels = cfg.channels
        self.block_size = cfg.blocksize
        self.threshold = cfg.threshold
        self.cfg = cfg
        self.max_rms = 0
        self.significant_channels = None

    def _compute_rms(self, block):
        """
        RMS amplitude per channel.
        block shape: (block_size, channels)
        returns: (channels,) float32
        """
        rms = np.sqrt(np.mean(block**2, axis=0))
        max = np.max(rms)
        if max > self.max_rms:
            self.max_rms = max
        return rms

    def _check_thresholds(self, rms):
        """
        Boolean mask of channels exceeding the threshold.
        returns: (channels,) bool
        """
        return rms > self.threshold

    def _compute_peak(self, block):
        """
        Peak absolute amplitude per channel.
        returns: (channels,) float32
        """
        return np.max(np.abs(block), axis=0)

    def _highpass_filter(self, block):
        """
        Apply a highpass Butterworth filter to the block.
        Returns the filtered block of the same shape.
        """
        b, a = signal.butter(
            self.cfg.filter_order,
            self.cfg.cutoff_freq / (self.fs * 0.5),
            "high",
        )
        return signal.lfilter(b, a, block, axis=0)

    def process(self):
        """
        Run all evaluations on a call queue.
        Returns a result dict passed to state.put_result().

        Extend this method with FFT, TDOA, beamforming, etc.
        """
        chunk = self._state.call_chunk.copy()
        time = self._state.call_time
        logger.debug("Processing call queue with %d samples", chunk.shape[0])

        time_delays = calc_multich_delays(
            chunk[:, self.significant_channels], self.cfg.fs
        )
        path_diff = time_delays * self.cfg.vsound
        locations = tristar_mellen_pachter(
            self._state.micxyz[self.significant_channels], path_diff
        )
        print(f"about tu push results, max rms = {self.max_rms}")
        self._state.put_result(locations, time)

        return True

    # ------------------------------------------------------------------
    # Processing loop
    # ------------------------------------------------------------------

    def run_loop(self):
        """
        Main processing loop — runs for the lifetime of the thread.

        get_audio() blocks for up to `timeout` seconds waiting for
        a new block. During that wait the GIL is released and other
        threads (GUI, audio callback) run freely — no busy-waiting.

        Returns when state.stop_event is set.
        """
        logger.info(
            "AudioProcessor loop started — fs=%d ch=%d blocksize=%d - callFlag %s",
            self.fs,
            self.channels,
            self.block_size,
            self._state.call_flag,
        )

        while not self._state.stop_event.is_set():

            # Blocking get with timeout — yields CPU while waiting.
            # Returns (None, None) on timeout so we loop back and
            # recheck stop_event without stalling indefinitely.
            block, timestamp = self._state.get_audio(timeout=0.1)

            if block is None:
                # Timeout: no new data yet, go back and wait again
                continue

            # here i shoud highpass filter the block.
            block = self._highpass_filter(block)
            rms = self._compute_rms(block)
            active_ch = self._check_thresholds(rms)

            if not self._state.call_flag:
                if np.any(active_ch):
                    self.significant_channels = np.where(active_ch)[0]
                    logger.info(
                        # "New call detected at %.2f s — active channels: %s",
                        timestamp,
                        self.significant_channels,
                    )
                    self._state.call_flag = True
                    self._state.call_time = timestamp
                    self._state.call_chunk = block

            elif self._state.call_flag == True:
                if (
                    np.any(active_ch)
                    or timestamp - self._state.call_time < self.cfg.call_time
                ):
                    self._state.call_chunk = np.append(
                        self._state.call_chunk, block, axis=0
                    )
                    chs = np.where(active_ch)[0]
                    self.significant_channels = np.union1d(
                        self.significant_channels, chs
                    )
                    logger.debug(
                        "Call updated at %.6f s — significant channels: %s - added channels: %s",
                        timestamp,
                        self.significant_channels,
                        chs,
                    )

                elif timestamp - self._state.call_time > self.cfg.call_time:
                    logger.info(
                        "Call ended at %.2f s — total duration: %.4f s - samples stored %i -  pushing results to queue",
                        timestamp,
                        timestamp - self._state.call_time,
                        self._state.call_chunk.shape[0],
                    )
                    if (
                        self.process()
                    ):  # is not detecting new calls until process is completed
                        self._state.call_chunk = np.ndarray([])
                        self._state.call_flag = False

        logger.info("AudioProcessor loop stopped")


# ---------------------------------------------------------------------------
# Thread entry point
# ---------------------------------------------------------------------------


def run(state, cfg):
    """
    Processing thread entry point — called once by the thread, never loops.
    Instantiates AudioProcessor and runs its loop until shutdown.
    """
    processor = AudioProcessor(state, cfg)
    processor.run_loop()
    logger.info("processing.run: exit")
