"""
audio_input.py

Provides two audio source classes with a unified interface:
  - RealtimeAudioSource: live capture via sounddevice/PortAudio
  - AudioFileSource: file playback with real-time pacing

Both write to SharedState.audio_queue via state.put_audio().
The processing thread reads from the same queue without knowing
which source is active.

Entry point for the audio thread: run(state, cfg)
"""

import time
import logging
import os
import numpy as np
import sounddevice as sd
import soundfile as sf

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def resolve_input_device(device_arg):
    """
    Resolve a device argument to a sounddevice device index.

    Accepts:
      - None        → system default input device
      - int / "3"   → device index
      - "str"       → partial name match (case-insensitive)

    Returns the resolved integer index, or None for system default.
    Raises ValueError if no matching input device is found.
    """
    if device_arg is None:
        return None

    devices = sd.query_devices()

    # Try numeric index first

    try:
        device_id = int(device_arg)
        dev = devices[device_id]
        if dev["max_input_channels"] <= 0:
            raise ValueError(f"Device {device_id} has no input channels")
        logger.info(
            "Resolved device index %d → %s (%d input channels)",
            device_id,
            dev["name"],
            dev["max_input_channels"],
        )
        return device_id
    except (ValueError, TypeError):
        pass

    # Fall back to partial name match
    device_arg_lower = str(device_arg).lower()
    for idx, dev in enumerate(devices):
        if device_arg_lower in dev["name"].lower() and dev["max_input_channels"] > 0:
            logger.info(
                "Resolved device name '%s' → index %d: %s (%d input channels)",
                device_arg,
                idx,
                dev["name"],
                dev["max_input_channels"],
            )
            return idx

    raise ValueError(f"Input device not found: '{device_arg}'")


# ---------------------------------------------------------------------------
# Realtime audio source
# ---------------------------------------------------------------------------


class RealtimeAudioSource:
    """
    Captures live audio from a hardware device via PortAudio.

    The PortAudio callback runs in a dedicated high-priority C thread.
    It must return as fast as possible — no blocking calls, no logging,
    no heavy computation. Only copy + put_nowait.

    Timing reference: time_info.inputBufferAdcTime (hardware ADC timestamp)
    is passed to state.put_audio() so the processing thread can use it
    for TDOA calculations without any additional synchronization.
    """

    def __init__(
        self,
        state,
        device=None,
        fs=192_000,
        channels=8,
        block_size=2048,
        # dtype="int16",
    ):
        self._state = state
        self.device = resolve_input_device(device)
        self.fs = fs
        self.channels = channels
        self.block_size = block_size
        self._stream = None
        self._chunknum = 0
        # self.dtype = dtype

    def _callback(self, indata, frames, time_info, status):
        """
        Called by PortAudio every block_size samples.
        Runs in PortAudio's internal C thread — keep this as fast as possible.

        indata shape: (block_size, channels), dtype float32
        time_info.inputBufferAdcTime: hardware ADC timestamp in seconds
        """
        if status:

            pass

        self._state.put_audio(indata.copy(), time_info.inputBufferAdcTime)
        self._chunknum += 1

    def start(self):
        """
        Opens the PortAudio stream and blocks the calling thread
        until state.stop_event is set.
        Equivalent role to AudioFileSource.start() — both block here.
        """
        logger.info(f"selected device: {sd.query_devices(self.device)}")
        try:
            self._stream = sd.InputStream(
                device=self.device,
                samplerate=self.fs,
                channels=self.channels,
                blocksize=self.block_size,
                # dtype=self.dtype,
                callback=self._callback,
                # latency="low",
            )
            self._stream.start()
            logger.info(
                "RealtimeAudioSource started — device=%s fs=%d ch=%d blocksize=%d",
                self.device,
                self.fs,
                self.channels,
                self.block_size,
            )

            # Block the audio thread here until shutdown is requested.
            # The callback keeps firing independently in PortAudio's thread.
            self._state.stop_event.wait()

        except sd.PortAudioError as e:
            logger.error("PortAudio error: %s", e)
            self._state.stop(__name__)  # propagate failure to all threads

    def stop(self):
        """Closes the PortAudio stream and releases resources."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            logger.info(
                "RealtimeAudioSource stopped — chunks delivered: %d  dropped: %d",
                self._chunknum,
                self._state.dropped_audio,
            )


# ---------------------------------------------------------------------------
# File audio source
# ---------------------------------------------------------------------------


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
        self.block_size = block_size
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
        self._block_duration = block_size / self.fs

        logger.info(
            "AudioFileSource ready — file=%s fs=%d samples=%d channels=%d",
            file_path,
            self.fs,
            self.audio.shape[0],
            self.audio.shape[1],
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
            range(0, total_samples - self.block_size, self.block_size)
        ):

            if self._state.stop_event.is_set():
                break

            block = self.audio[start : start + self.block_size, :]

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


# ---------------------------------------------------------------------------
# Thread entry point
# ---------------------------------------------------------------------------


def run(state, cfg):
    """
    Audio thread entry point — called once by the thread, never loops.

    Instantiates the correct source based on cfg.mode,
    calls source.start() which blocks until shutdown,
    then calls source.stop() for cleanup.
    """
    if cfg.mode == "realtime":
        source = RealtimeAudioSource(
            state=state,
            device=cfg.device,
            fs=cfg.fs,
            channels=cfg.channels,
            block_size=cfg.blocksize,
            # dtype=cfg.dtype,
        )
    elif cfg.mode == "audiofile":
        source = AudioFileSource(
            state=state,
            file_path=cfg.file,
            block_size=cfg.blocksize,
            loaded_durn=getattr(cfg, "loaded_durn", None),
        )
    else:
        raise ValueError(
            f"Unknown audio mode: '{cfg.mode}' — expected 'realtime' or 'audiofile'"
        )
    while not state.gui_running_flag:
        time.sleep(1)
        if state.stop_event.isSet():
            logger.info("audio stream never started - exiting audio thread")
            return
    state.start()
    source.start()  # blocks until stop_event or end of file
    source.stop()  # cleanup - nothing is actually done by now

    logger.info("audio_input.run: exit")
