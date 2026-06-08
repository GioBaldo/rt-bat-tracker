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

import logging
import queue
import signal
import sys
import threading

# ---------------------------------------------------------------------------
# Logging — configura prima di importare i moduli applicativi
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(threadName)-20s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import moduli applicativi
# ---------------------------------------------------------------------------
import rt_bat_tracker.audio.audio_input as audio_input
import rt_bat_tracker.tracking.beta_processing as processing
import rt_bat_tracker.GUI.gui_update as gui_update

from PyQt5.QtWidgets import QApplication

# ---------------------------------------------------------------------------
# Costanti code
# ---------------------------------------------------------------------------
AUDIO_QUEUE_MAXSIZE = 4  # blocchi audio in attesa di processing
# piccolo = bassa latenza, ma processing deve stare al passo
RESULT_QUEUE_MAXSIZE = 2  # risultati in attesa della GUI
# la GUI non ha mai più di 2 frame di ritardo


def main():
    # --- Code condivise thread-safe ---
    audio_queue = queue.Queue(maxsize=AUDIO_QUEUE_MAXSIZE)
    result_queue = queue.Queue(maxsize=RESULT_QUEUE_MAXSIZE)

    # --- Evento di stop condiviso tra tutti i thread ---
    # Quando viene settato, ogni thread termina il proprio loop al prossimo giro.
    stop_event = threading.Event()

    # --- Signal handler per SIGINT (Ctrl-C) e SIGTERM ---
    def _handle_signal(signum, frame):
        logger.info("Segnale %s ricevuto — shutdown in corso...", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # --- Thread audio (priorità alta, daemon) ---
    audio_thread = threading.Thread(
        target=audio_input.run,
        args=(audio_queue, stop_event),
        name="AudioInput",
        daemon=True,  # terminato automaticamente se il main thread muore
    )

    # --- Thread processing (daemon) ---
    proc_thread = threading.Thread(
        target=processing.run,
        args=(audio_queue, result_queue, stop_event),
        name="Processing",
        daemon=True,
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
    app = QApplication(sys.argv)

    # Su RPi4: se usi eglfs forza il platform plugin corretto
    # export QT_QPA_PLATFORM=eglfs   (oppure xcb se hai X11)

    logger.info("GUI avviata nel main thread")
    gui_update.run(result_queue, stop_event, app=app)

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
