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
import hydra
import math
from omegaconf import DictConfig, OmegaConf
from queue import Empty
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
    micLayout_name = audioFileName.replace(".wav", ".csv")
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

    def get_xyz_tristar():
        nonlocal chunknum

        audiochunk, chunknum = audio_source.get_block()

        if audiochunk is None:
            return np.array([])

        channel_rms = np.array(
            [calc_rms(audiochunk[:, each]) for each in range(audiochunk.shape[1])]
        )
        thrsold_exceeded = np.array(
            ["X" if rms > cfg.threshold else "_" for rms in channel_rms]
        )

        if np.all(channel_rms > cfg.threshold):
            delays = calc_multich_delays(audiochunk, ba_filt=highpass_coeffs, fs=fs)
            di = delays * cfg.vsound
            solns = tristar_mellen_pachter(micxyz, di)

            if len(solns) > 0:
                source_solutions.put((solns, chunknum))
                return solns
        print(
            f"Processing chunk number {chunknum} with shape {audiochunk.shape} threshold exceeded: {thrsold_exceeded} delays [ms]: {np.around(delays*1000,3) if 'delays' in locals() else 'N/A'}"
        )
        return np.array([])

    app = pg.mkQApp("Realtime angle-of-arrival plot")

    w = gl.GLViewWidget()
    w.show()
    w.setWindowTitle("Realtime angle-of-arrival plot")
    w.setCameraPosition(distance=25)

    g = gl.GLGridItem()
    w.addItem(g)

    mic_plot = gl.GLScatterPlotItem(pos=micxyz, color=(1, 0, 0, 1), size=10)
    w.addItem(mic_plot)

    all_sources = [np.array([100, 100, 100])]
    source_plot = gl.GLScatterPlotItem(pos=all_sources, color=(1, 1, 0, 1), size=10)
    w.addItem(source_plot)

    camdistance = 20
    updatenum = 1

    w.setCameraParams(distance=camdistance, azimuth=0)
    w.grabFramebuffer().save(f"{projPaths.png_dir}/only_array.png")

    def update():
        nonlocal updatenum, camdistance

        out = get_xyz_tristar()

        if out is None or len(out) == 0:
            return

        updatenum += 1

        for each in out:
            all_sources.append(each)

        source_plot.setData(pos=all_sources)
        w.grabFramebuffer().save(f"{projPaths.png_dir}/xyzSolution_{chunknum}.png")

        if w.cameraParams()["azimuth"] >= -90:
            w.orbit(-2, -0.25)
        else:
            camdistance += 0.25
            w.setCameraParams(distance=camdistance)
            w.orbit(0, 2)

    t = QtCore.QTimer()
    t.timeout.connect(update)
    t.start(10)

    pg.exec()

    audio_source.stop()

    image_files = natsort.natsorted(glob.glob(f"{projPaths.png_dir}/xyzSolution_*.png"))

    if len(image_files) > 0:
        from PIL import Image, ImageDraw

        frames = []

        for img in image_files:
            new_frame = Image.open(img)
            draw = ImageDraw.Draw(new_frame)

            if not source_solutions.empty():
                solns, chunk_id = source_solutions.get()
                draw.text((0, 0), f"Buffer number {chunk_id}", (255, 0, 255))

            frames.append(new_frame)

        frames[0].save(
            "png_to_gif.gif",
            format="GIF",
            append_images=frames[1:],
            save_all=True,
            duration=80,
            loop=0,
        )

        # for each in image_files:
        # os.remove(each)
    else:
        print("No solution images found, skipping gif creation.")


if __name__ == "__main__":
    main()
