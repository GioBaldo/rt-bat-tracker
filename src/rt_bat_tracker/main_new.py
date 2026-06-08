"""bat-tracker pipeline for real-time tracking of bats using microphone arrays.

Returns:
    displays the 3d position of the sound source in real time, and saves a gif of the tracking results.
"""

import numpy as np
import sys
import argparse
import os
import glob
import natsort
import time
import queue
import math
import threading
from omegaconf import DictConfig, OmegaConf
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore
from scipy import signal
import sounddevice as sd
import soundfile as sf
from rt_bat_tracker.utils.cli import parse_args, list_input_devices
from rt_bat_tracker.utils.paths import get_project_paths
from rt_bat_tracker.tracking.common_functions import calc_rms, calc_multich_delays
from rt_bat_tracker.tracking.localisation_mpr2003 import tristar_mellen_pachter
from rt_bat_tracker.audio.sources import AudioFileSource, RealtimeAudioSource
from rt_bat_tracker.audio.detector import AudioBlockMonitor
from rt_bat_tracker.GUI.gui_functions import GUIUpdate
from scipy.spatial.transform import Rotation

# import project paths
projPaths = get_project_paths()


# @hydra.main(
#    version_base=None,
#    config_path=str(projPaths.config_dir),
#    config_name="config",  # usually without .yaml
# )
def main():
    cfg = OmegaConf.load(projPaths.config_dir / "config.yaml")

    args = parse_args(
        projPaths.audio_dir,
        default_file=cfg.default_file,
        default_device=cfg.default_device,
    )

    if args.list_devices:
        list_input_devices()
        return

    audioFileName = os.path.basename(args.file)
    micLayout_name = cfg.default_layout
    micLayout_path = os.path.join(projPaths.mic_layout_dir, micLayout_name)

    print(f"Loading mic layout from: {micLayout_name}")
    micxyz = np.loadtxt(micLayout_path, delimiter=",")
    num_mics = micxyz.shape[0]
    print(f"mics used: {num_mics}")

    if args.mode == "audiofile":
        audio_source = AudioFileSource(
            file_path=args.file,
            block_size=cfg.blocksize,
            loaded_durn=9,
        )

    elif args.mode == "realtime":
        audio_source = RealtimeAudioSource(
            device=args.device,
            fs=cfg.fs,
            channels=num_mics,
            block_size=cfg.blocksize,
        )

    audio_source.start()

    fs = audio_source.fs
    if fs != cfg.fs:
        print(
            f"Warning: audio source sample rate {fs} does not match config sample rate {cfg.fs}"
        )

    highpass_coeffs = signal.butter(
        cfg.filter_order, cfg.cutoff_freq / (fs * 0.5), "high"
    )
    source_solutions = queue.Queue()
    chunknum = 0

    gui_test = GUIUpdate(0.02)
    gui_test.start()
    monitor = AudioBlockMonitor(source=audio_source)
    monitor.start()
    # monitor.thread.join()


if __name__ == "__main__":
    main()
