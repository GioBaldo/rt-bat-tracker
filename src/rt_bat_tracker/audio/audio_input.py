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
from rt_bat_tracker.audio.alsa_source import AlsaAudioSource 
from rt_bat_tracker.audio.portaudio_source import PortAudioSource
from rt_bat_tracker.audio.audiofile_source import AudioFileSource

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# ---------------------------------------------------------------------------
# Audio thread entry point
# ---------------------------------------------------------------------------

def run(state, cfg):
    """
    Audio thread entry point — called once by the thread, never loops.

    Instantiates the correct source based on cfg.mode,
    calls source.start() which blocks until shutdown,
    then calls source.stop() for cleanup.
    """

    # source selector
    if cfg.mode == "portAudio":
        source = PortAudioSource(
            state=state,
            device=cfg.device,
            fs=cfg.fs,
            channels=cfg.channels,
            block_size=cfg.blocksize,
            dtype=cfg.dtype,
        )

    elif cfg.mode == "alsa":
        source = AlsaAudioSource(state)#, device=cfg.device, fs=cfg.fs, channels=cfg.channels, blocksize=cfg.blocksize)

    elif cfg.mode == "audiofile":
        source = AudioFileSource(
            state=state,
            file_path=cfg.file,
            block_size=cfg.blocksize,
            loaded_durn=getattr(cfg, "loaded_durn", None),
        )
    else:
        raise ValueError(
            f"Unknown audio mode: '{cfg.mode}' — expected 'alsa' , 'portAudio' or 'audiofile'"
        )

    # waiting gor gui to be loaded
    while not state.gui_running_flag:
        time.sleep(1)
        if state.stop_event.isSet():
            logger.info("audio stream never started - exiting audio thread")
            return

    logger.info(
        f"stream start soon with dev[{source.device}], fs[{source.fs}], channels[{source.channels}], blocksize[{source.blocksize}]"
    )

    # start source
    time.sleep(1)
    state.start()
    e = source.start()  # blocks until stop_event or end of file
    if e is not None:   # if e is None means that the stop wa requested by another thread
        state.stop(e)   # otherwise e is the name of the source which requested the stop
    
    logger.info("audio_input.run: exit")
