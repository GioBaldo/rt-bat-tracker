import sounddevice as sd
import time
import logging
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# ---------------------------------------------------------------------------
# Realtime audio source
# ---------------------------------------------------------------------------
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



class PortAudioSource:
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
        device=21,
        fs=192000,
        channels=8,
        block_size=1024,
        dtype="float32",
    ):
        self._state = state
        self.device = resolve_input_device(device)
        self.fs = fs
        self.channels = channels
        self.block_size = block_size
        self._stream = None
        self._chunknum = 0
        self.dtype = dtype

    def _callback(self, indata, frames, time_info, status):
        """
        Called by PortAudio every block_size samples.
        Runs in PortAudio's internal C thread — keep this as fast as possible.

        indata shape: (block_size, channels), dtype float32
        time_info.inputBufferAdcTime: hardware ADC timestamp in seconds
        """
        if status:

            pass
        logger.debug(
            f"AUDIO - Input data type: {type(indata[0][0])} - channels, blocksize: {np.shape(indata)} - maxval = {np.max(np.abs(indata))}"
        )
        self._state.put_audio(indata.copy(), 0)
        self._chunknum += 1

    def start(self):
        """
        Opens the PortAudio stream and blocks the calling thread
        until state.stop_event is set.
        Equivalent role to AudioFileSource.start() — both block here.
        """
        logger.debug(f"sounddevice specd: {sd._libname}")
        logger.info(f"selected device: {sd.query_devices(self.device)}")
        logger.debug(f"default device: {sd.default.device}")
        logger.debug(f"default fs: {sd.default.samplerate}")
        for i, api in enumerate(sd.query_hostapis()):
            logger.info(f"host api: {i}, name: {api['name']}")
        for dtype in ["float32", "int32", "int24", "int16"]:
            try:
                sd.check_input_settings(
                    device=self.device,
                    channels=self.channels,
                    samplerate=self.fs,
                    dtype=dtype,
                )
                logger.info(f"OK, {dtype}")
                self.dtype = dtype
                break
            except Exception as e:
                print("FAIL", dtype, e)
        # logger.info(
        #     f"trying to open {sd.query_devices(self.device)['name']} with fs = {self.fs}, blocksize = {self.block_size}, dtype = {self.dtype}"
        # )
        try:
            self._stream = sd.InputStream(
                device=self.device,
                samplerate=self.fs,
                channels=self.channels,
                blocksize=self.block_size,
                dtype=self.dtype,
                callback=self._callback,
                # latency="low",
            )
            with self._stream:
                # logger.info(
                #     "RealtimeAudioSource started — device=%s fs=%d ch=%d blocksize=%d",
                #     self.device,
                #     self.fs,
                #     self.channels,
                #     self.block_size,
                # )

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
