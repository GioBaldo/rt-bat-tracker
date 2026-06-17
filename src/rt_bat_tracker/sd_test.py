import os

os.environ["SD_ENABLE_ASIO"] = "1"
import numpy as np
import sounddevice as sd
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
device = 21

print(sd._libname)
# print(sd.query_hostapis())
sd.default.device = device
sd.default.samplerate = 44100
# sd.default.samplerate = 192000
print(sd.query_devices(device))


duration = 5  # seconds


def callback(indata, frames, time, status):
    if status:
        print(status)
    print(np.shape(indata))


def start():
    print(f"sounddevice specd: {sd._libname}")
    print(f"selected device: {sd.query_devices(device)}")
    print(f"default device: {sd.default.device}")
    print(f"default fs: {sd.default.samplerate}")
    sd.sleep(2000)
    try:
        stream = sd.InputStream(callback=callback)
        with stream:
            sd.sleep(1000)
    except KeyboardInterrupt:
        print("InputStream stopped by user")
    except sd.PortAudioError as e:
        print(("PortAudio error: %s", e))


start()
