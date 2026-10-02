"""
processing.py

Reads audio blocks from SharedState.audio_queue and runs
signal evaluation on each block.

The processing loop uses a blocking get() with timeout so the thread
yields CPU to other threads while waiting for new data — no busy-waiting.
Results are written to SharedState.result_queue via state.put_result().

Entry point for the processing thread: run(state, cfg)
"""

from dataclasses import dataclass, asdict, field
import logging
import queue
import time

import numpy as np
from scipy import signal

from rt_bat_tracker.tracking.localisation_mpr2003 import tristar_mellen_pachter
from rt_bat_tracker.tracking.zero_sum_localisation import ZeroSumLocalizer
from rt_bat_tracker.tracking.common_functions import (
    calc_rms,
    calc_multich_delays,
    calc_delay,
)

# import librosa
# from scipy import signal

logger = logging.getLogger("PROC")
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Processor class
# ---------------------------------------------------------------------------


@dataclass
class LocAnalytics:
    """
    Data class to hold analytics for localization results.
    """

    start_time: float = 0.0
    call_duration: float = 0.0
    max_rms: float = 0.0
    significant_channels: list = field(default_factory=list)
    processing_duration: float = 0.0
    chunk_size: int = 0
    cross_correlations: list = field(default_factory=list)
    channel_pairs: list = field(default_factory=list)
    TDOA: list = field(default_factory=list)
    TDOA_sum: float = 0.0


##THREAD ENTRY POINT##


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
        self.loc_method = state.loc_method
        self.max_speed = 5.0  # m/s
        self.last_valid_loc = None
        self.last_call_lime = None
        self.localizer = ZeroSumLocalizer(cfg=self.cfg)
        self.blocks_received = 0
        self.max_rms_channel = None
        self.significant_channels = None

    def _compute_rms(self, block):
        """
        RMS amplitude per channel.
        block shape: (block_size, channels)
        returns: (channels,) float32
        """
        sq_sig = block**2
        peak_idx = np.argmax(sq_sig, axis=0)
        mean_sq = np.mean(sq_sig, axis=0)
        rms = np.sqrt(mean_sq)
        max_v = np.max(rms)
        self._state.EMA_rms = self._state.EMA_rms + 0.5 * (max_v - self._state.EMA_rms)
        self.blocks_received += 1
        self._state.avg_rms = (
            self._state.avg_rms * (self.blocks_received - 1) + max_v
        ) / self.blocks_received
        if max_v > self._state.max_rms:
            self._state.max_rms = max_v
            self.max_rms_channel = np.where(rms == max_v)[0][0]
        return rms, max_v, peak_idx

    def _check_thresholds(self, rms):
        """
        Boolean mask of channels exceeding the threshold.
        returns: (channels,) bool
        """
        return rms > self._state.threshold

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
        timestamp = self._state.call_time
        analytics = LocAnalytics(
            start_time=time.perf_counter_ns(), chunk_size=chunk.shape[0]
        )
        logger.debug(
            f"Processing call chunk with {chunk.shape} samples, array type: {type(chunk)}, sample type: {type(chunk[0][0])}"
        )

        ## DIFFERENT LOCALIZATION METHODS ##
        # DEFAULT
        if self.loc_method == "default_mpr":

            time_delays = calc_multich_delays(
                chunk[:, self.significant_channels], self.fs
            )

            path_diff = (
                time_delays * self.cfg.vsound
            )  # Localization compudet on Range Difference! [meters]
            locations = tristar_mellen_pachter(
                self._state.micxyz[self.significant_channels],
                path_diff,
                self._state.normal_vector,
            )

        # ZERO-SUM CHECK
        elif self.loc_method == "zero_sum":
            path_diff, selected_mics, analytics = self.localizer.get_zerosum_delays(
                chunk=chunk,
                fs=self.fs,
                ch_to_keep=6,
                significant_channels=self.significant_channels.tolist(),
            )

            locations = tristar_mellen_pachter(
                self._state.micxyz[selected_mics],
                path_diff,
                self._state.normal_vector,
            )

        # SPEED CONSISTENCY CHECK
        elif self.loc_method == "speed_consistency":
            time_delays = calc_multich_delays(
                chunk[:, self.significant_channels], self.fs
            )

            path_diff = (
                time_delays * self.cfg.vsound
            )  # Localization compudet on Range Difference! [meters]
            locations = tristar_mellen_pachter(
                self._state.micxyz[self.significant_channels],
                path_diff,
                self._state.normal_vector,
            )
            if self.last_valid_loc is not None:
                speed = np.linalg.norm(
                    np.array(locations) - np.array(self.last_valid_loc)
                ) / (self._state.call_time - self._state.last_call_time)
                print(f"Speed: {speed:.2f} m/s")
                if speed > self.max_speed:
                    logger.warning(
                        f"Unrealistic speed detected: {speed:.2f} m/s. Discarding location."
                    )
                    locations = None

            if locations is not None and len(locations) > 0:
                self.last_valid_loc = locations
                self._state.last_call_time = self._state.call_time

        logger.info(
            f"about to push results, max rms = {self._state.max_rms} - locations: {locations} - dtype: {type(chunk[0][0])}"
        )
        if locations is None and len(time_delays) != 0:
            logger.error("ERROR! TRYING TO COMPUTE TDOA WITH THE WRONG MIC LAYOUT")

        analytics.processing_duration = time.perf_counter_ns() - analytics.start_time
        print(f"Processing duration: {analytics.processing_duration / 1000000:.4f} ms")

        if locations is not None:
            self._state.put_result(locations, timestamp)

        return True

    ##PROCESSING LOOP##

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

            # put HP block in the event audio queue for later saving (all channels)
            self._state.write_wav_buffer(block)

            # remove unused channels
            block = block[:, : self._state.micxyz.shape[0]]

            rms, max_channel, peak_idx = self._compute_rms(block)

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
                        logger.info(
                            "Call updated at %.3f s — significant channels: %s - active channels: %s",
                            timestamp,
                            self.significant_channels,
                            chs,
                        )
                        continue

                logger.debug(
                    f"call ended: active channels: {self.significant_channels}, call duration: {timestamp - self._state.call_time:.4f} s, samples stored: {self._state.call_chunk.shape[0]}"
                )

                logger.info(f"significant channels: {self.significant_channels}")
                if self.significant_channels.size > 3:
                    self.process()

                self._state.call_chunk = np.ndarray([])
                self._state.call_flag = False

        logger.info("AudioProcessor loop stopped")
        return
