import sounddevice as sd
import argparse
import glob
import os
import sys


def list_input_devices():
    devices = sd.query_devices()
    default_input = (
        sd.default.device[0]
        if isinstance(sd.default.device, (list, tuple))
        else sd.default.device
    )
    print("Available audio devices:")
    for idx, dev in enumerate(devices):
        if dev["max_output_channels"] > 0:
            default_tag = " (default)" if idx == default_input else ""
            print(
                f"{idx}: {dev['name']} - outputs={dev['max_output_channels']}{default_tag}"
            )


def parse_args(
    audioFiles_dir, default_file="single_bat_1234.wav", default_device=None, argv=None
):
    parser = argparse.ArgumentParser(
        description="Realtime bat tracking from audio input or file."
    )

    wav_files = glob.glob(os.path.join(audioFiles_dir, "*.wav"))
    wav_names = [os.path.basename(f) for f in wav_files]

    if default_file not in wav_names and len(wav_names) > 0:
        default_file = wav_names[0]

    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio input devices and exit",
    )

    parser.add_argument(
        "--device", default=default_device, help="Input device index or name to use"
    )

    parser.add_argument(
        "--loopback",
        action="store_true",
        help="Enable WASAPI loopback capture on Windows",
    )

    parser.add_argument(
        "--mode",
        default="audiofile",
        choices=["realtime", "audiofile"],
        help="Capture mode, default audiofile",
    )

    parser.add_argument(
        "--file",
        default=default_file,
        choices=wav_names,
        help="Select file to use in audiofile mode",
    )

    args = parser.parse_args(argv)

    if args.file is not None:
        args.file = os.path.join(audioFiles_dir, args.file)

    return args
