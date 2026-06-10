"""
main.py
Orchestrazione dell'applicazione.
Il main NON contiene logica audio, processing o GUI.
Si occupa solo di:
  - creare le code condivise
  - avviare i thread
  - gestire SIGINT / SIGTERM per uno shutdown pulito
  - attendere la terminazione ordinata

NOTA: la GUI (Qt) gira nel main thread perché Qt lo richiede su molte
piattaforme (in particolare su RPi4 con eglfs/xcb).
I thread audio e processing sono daemon thread — vengono terminati
automaticamente se il processo principale muore.
"""

# %% Imports
import logging
import os

os.environ["PYQTGRAPH_QT_LIB"] = "PyQt5"
import queue
import signal
import sys
import numpy as np
import threading
from omegaconf import OmegaConf
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore
import time
import sounddevice as sd

# ---------------------------------------------------------------------------
# Import modules
# ---------------------------------------------------------------------------
from rt_bat_tracker.utils.cli import parse_args, list_input_devices
from rt_bat_tracker.utils.paths import get_project_paths
import rt_bat_tracker.audio.audio_input as audio_input
import rt_bat_tracker.tracking.beta_processing as processing
import rt_bat_tracker.GUI.gui_update as gui
from rt_bat_tracker.utils.dataClass import SharedState
from rt_bat_tracker.utils.json_formatter import JsonFormatter


from PyQt5.QtWidgets import QApplication

# import project paths
projPaths = get_project_paths()

# set up logging
# main.py
logger = logging.getLogger()  # no name = root
logger.setLevel(logging.ERROR)

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(
    logging.Formatter("%(threadName)s - %(levelname)s - %(message)s")
)

file_handler = logging.FileHandler(projPaths.results_dir / "app.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(JsonFormatter())

logger.addHandler(stream_handler)
logger.addHandler(file_handler)

# ----------------------------------------------------------------------------
# Main function


def main():

    cfg = OmegaConf.load(projPaths.config_dir / "config.yaml")
    global micxyz
    sd.default.device = (cfg.default_device, None)

    args = parse_args(
        projPaths.audio_dir,
        default_file=cfg.default_file,
        default_device=cfg.default_device,
    )

    if args.list_devices:
        list_input_devices()
        return

    # pass arg inputs to the cfg dictionary
    cfg.mode = args.mode
    cfg.file = args.file
    cfg.device = args.device
    cfg.micLayout_path = str(os.path.join(projPaths.mic_layout_dir, cfg.default_layout))
    cfg.GUIpath = str(os.path.join(projPaths.gui_dir, "GUI_Layout.ui"))

    logger.info("Modalità di acquisizione: %s", cfg.mode)

    # initialize shared state class
    state = SharedState(cfg)

    # --- Signal handler per SIGINT (Ctrl-C) e SIGTERM ---
    def _handle_signal(signum, frame):
        logger.info("Segnale %s ricevuto — shutdown in corso...", signum)
        state.stop()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # --- Thread audio (priorità alta, daemon) ---
    audio_thread = threading.Thread(
        target=audio_input.run,
        args=(state, cfg),
        name="AudioInput",
        daemon=False,  # terminato automaticamente se il main thread muore
    )

    # --- Thread processing (daemon) ---
    proc_thread = threading.Thread(
        target=processing.run,
        args=(state, cfg),
        name="Processing",
        daemon=False,
    )

    # --- Avvio thread ---
    audio_thread.start()
    logger.info("Thread AudioInput avviato  (tid=%d)", audio_thread.ident)

    proc_thread.start()
    logger.info("Thread Processing avviato  (tid=%d)", proc_thread.ident)

    # --- GUI nel main thread (requisito Qt) ---
    # QApplication deve essere creata nel main thread.
    # gui_update.run() blocca qui finché la finestra non viene chiusa,
    # dopodiché setta stop_event tramite app.aboutToQuit.
    logger.info("loading gui... ")
    state.gui_t_start = time.time()
    gui.run(state, cfg)
    # app.exec()
    # Su RPi4: se usi eglfs forza il platform plugin corretto
    # export QT_QPA_PLATFORM=eglfs   (oppure xcb se hai X11)

    logger.info("GUI avviata nel main thread")
    # gui_update.run(state, app=app)

    # --- Shutdown: la GUI è uscita, stop_event è già settato ---
    logger.info("GUI chiusa — attendo terminazione thread...")

    # join con timeout: se un thread non risponde entro 3s, proseguiamo comunque
    for t, name in [(audio_thread, "AudioInput"), (proc_thread, "Processing")]:
        t.join(timeout=3.0)
        if t.is_alive():
            logger.warning("%s non terminato entro il timeout", name)
        else:
            logger.info("%s terminato correttamente", name)

    logger.info("Applicazione chiusa.")


if __name__ == "__main__":
    main()
