import queue
import soundfile as sf
import sounddevice as sd
import numpy as np
import time


def resolve_input_device(device_arg):
    """
    device_arg can be:
    - None
    - integer index as string, e.g. "3"
    - device name or partial name
    """

    if device_arg is None:
        return None  # use system default input device

    devices = sd.query_devices()

    # Case 1: numeric index
    try:
        device_id = int(device_arg)
        dev = devices[device_id]

        if dev["max_input_channels"] <= 0:
            raise ValueError(f"Device {device_id} has no input channels")
        print(
            f'Resolved device index {device_id} to: {dev["name"]}, input channels: {dev["max_input_channels"]}'
        )
        return device_id

    except ValueError:
        pass

    # Case 2: name / partial name
    device_arg_lower = device_arg.lower()

    for idx, dev in enumerate(devices):
        if device_arg_lower in dev["name"].lower() and dev["max_input_channels"] > 0:
            print(
                f"Resolved device '{device_arg}' to index {idx}: {dev['name']}, input channels: {dev['max_input_channels']}"
            )
            return idx

    raise ValueError(f"Input device not found: {device_arg}")


# used only in case of audiofile source
class AudioFileSource:
    def __init__(self, file_path, block_size=2048, loaded_durn=9):
        self.file_path = file_path
        self.block_size = block_size

        self.fs = sf.info(file_path).samplerate
        self.audio, self.fs = sf.read(file_path, stop=int(self.fs * loaded_durn))

        self.starts = range(0, self.audio.shape[0], self.block_size)
        self.index = 0

    def start(self):
        pass

    def get_block(self):
        if self.index >= len(self.audio):
            return None, None

        start = self.index
        chunk = self.audio[start : start + self.block_size, :]
        chunknum = self.index // self.block_size
        self.index += self.block_size

        return chunk, chunknum

    def get_blocksize(self):
        return self.block_size

    def stop(self):
        pass


class RealtimeAudioSource:
    def __init__(self, device=None, fs=192000, channels=8, block_size=2048):
        self.device = resolve_input_device(device)
        self.fs = fs
        self.channels = channels
        self.block_size = block_size
        self.queue = queue.Queue(maxsize=20)
        self.stream = None
        self.chunknum = 0

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(status)

        try:
            self.queue.put_nowait((indata.copy(), self.chunknum))
            self.chunknum += 1
        except queue.Full:
            pass

    def start(self):
        self.stream = sd.InputStream(
            device=self.device,
            samplerate=self.fs,
            channels=self.channels,
            blocksize=self.block_size,
            dtype="float32",
            callback=self._callback,
        )
        self.stream.start()

    def get_block(self):
        return self.queue.get()

    def get_blocksize(self):
        return self.block_size

    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
