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
import time

import numpy as np
from scipy import signal

from rt_bat_tracker.tracking.localisation_mpr2003 import tristar_mellen_pachter
from rt_bat_tracker.tracking.common_functions import calc_rms, calc_multich_delays

# import librosa
# from scipy import signal

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
        self.cfg = cfg

        self.blocks_received = 0
        self.max_rms_channel = None
        self.significant_channels = None

    def _compute_rms(self, block):
        """
        RMS amplitude per channel.
        block shape: (block_size, channels)
        returns: (channels,) float32
        """
        rms = np.sqrt(np.mean(block**2, axis=0))
        max = np.max(rms)
        self.blocks_received += 1
        self._state.avg_rms = (
            self._state.avg_rms * (self.blocks_received - 1) + max
        ) / self.blocks_received
        if max > self._state.max_rms:
            self._state.max_rms = max
            self.max_rms_channel = np.where(rms == max)[0][0]
        return rms

    def _check_thresholds(self, rms):
        """
        Boolean mask of channels exceeding the threshold.
        returns: (channels,) bool
        """
        return rms > self._state.threshold

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
        logger.debug(
            f"Processing call chunk with {chunk.shape} samples, array type: {type(chunk)}, sample type: {type(chunk[0][0])}"
        )

        time_delays = calc_multich_delays(
            chunk[:, self.significant_channels], self.cfg.fs
        )

        path_diff = time_delays * self.cfg.vsound
        locations = tristar_mellen_pachter(
            self._state.micxyz[self.significant_channels], path_diff
        )
        logger.info(
            f"about to push results, max rms = {self._state.max_rms} - locations: {locations} - dtype: {type(chunk[0][0])}"
        )
        if len(locations) == 0 & len(time_delays) != 0:
            logger.error("ERROR! TRYING TO COMPUTE TDOA WITH THE WRONG MIC LAYOUT")

        self._state.put_result(locations, time)

        return True

    # ------------------------------------------------------------------
    # Processing loop
    # ------------------------------------------------------------------

    def run_loop(self):
        """
        Main processing loop — runs for the lifetime of the thread.

        Description:
        At each cycle reads block, timeastamp from the audio queue,
        then evaluates RMS values for all the channels and checks threshold.
        RMS are evaluated after a highpass filter is applied to the block.
        As soon as a channel exceeds the threshold a new call is set
        and a call_chunk is updated in order to have a 5 - 10 ms audio
        chunk to be processed.

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
                logger.debug("block is None")
                continue

            # highpassfilter to remove useless low end
            block = self._highpass_filter(block)
            # here data are implicitly converted from np.float32 to np.float64

            # put HP block in the event audio queue for later saving
            self._state.write_wav_buffer(block)

            rms = self._compute_rms(block)
            logger.debug(
                f"channel rms: {np.max(rms)} on channel {np.where(rms == np.max(rms))[0][0]} "
            )

            # compares rms values with thresholds to identify active channels
            active_ch = self._check_thresholds(rms)

            if not self._state.call_flag:
                if np.any(active_ch):
                    self.significant_channels = np.where(active_ch)[0]
                    logger.info(
                        "New call detected at %.3f s — active channels: %s, rms: %f",
                        timestamp,
                        self.significant_channels,
                        np.max(rms),
                    )
                    self._state.call_flag = True
                    self._state.call_time = timestamp
                    self._state.call_chunk = block
                    continue

            elif self._state.call_flag == True:
                if (timestamp - self._state.call_time) < self.cfg.MAX_CALL_DURATION:
                    if (
                        np.any(active_ch)
                        or timestamp - self._state.call_time
                        < self.cfg.MIN_CALL_DURATION
                    ):
                        self._state.call_chunk = np.append(
                            self._state.call_chunk, block, axis=0
                        )
                        chs = np.where(active_ch)[0]
                        self.significant_channels = np.union1d(
                            self.significant_channels, chs
                        )
                        logger.debug(
                            "Call updated at %.3f s — significant channels: %s - added channels: %s",
                            timestamp,
                            self.significant_channels,
                            chs,
                        )
                        continue

                    # elif timestamp - self._state.call_time > self.cfg.MIN_CALL_DURATION:
                    #     logger.info(
                    #         "Call ended at %.2f s — total duration: %.4f s - samples stored %i -  pushing results to queue",
                    #         timestamp,
                    #         timestamp - self._state.call_time,
                    #         self._state.call_chunk.shape[0],
                    #     )
                logger.debug(
                    f"call ended: active channels: {self.significant_channels}, call duration: {timestamp - self._state.call_time:.4f} s, samples stored: {self._state.call_chunk.shape[0]}"
                )
                if self.significant_channels.size > 3:
                    self.process()

                self._state.call_chunk = np.ndarray([])
                self._state.call_flag = False

        logger.info("AudioProcessor loop stopped")
        return


# ---------------------------------------------------------------------------
# Thread entry point
# ---------------------------------------------------------------------------


def run(state, cfg):
    """
    Processing thread entry point — called once by the thread, never loops.
    Instantiates AudioProcessor and runs its loop until shutdown.
    """
    while not state.gui_running_flag:
        time.sleep(0.1)
        if state.stop_event.isSet():
            logger.info("processing loop never started - exiting processing thread")
            return
    processor = AudioProcessor(state, cfg)
    processor.run_loop()
    logger.info(
        "processing.run: exit . STATS[max_rms: %.4f, avg_rms: %.4f, blocks_received: %d]",
        state.max_rms,
        state.avg_rms,
        processor.blocks_received,
    )
    return
