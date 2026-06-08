"""
processing.py
Consuma blocchi audio dalla audio_queue, esegue:
  1. FFT con finestra più grande del blocksize (overlap-add su buffer circolare)
  2. TDOA tra i canali (GCC-PHAT semplificato)
  3. Stima posizione 3D (placeholder — da sostituire con il tuo array geometry)
Mette i risultati in result_queue per la GUI.
"""

import numpy as np
import logging
import queue

logger = logging.getLogger(__name__)

# --- Parametri processing ---
SAMPLERATE = 192_000
CHANNELS = 8
BLOCKSIZE = 1024
FFT_WINDOW = 4096  # finestra FFT più grande del blocksize
# → migliore risoluzione frequenziale (~47 Hz/bin a 192kHz)
OVERLAP = FFT_WINDOW - BLOCKSIZE  # sample di overlap tra finestre successive

# GCC-PHAT: canale di riferimento per il calcolo dei delay
REF_CHANNEL = 0


def _gcc_phat(sig_ref, sig_other, fs):
    """
    Generalized Cross-Correlation with Phase Transform.
    Restituisce il delay stimato in secondi tra sig_ref e sig_other.
    """
    n = len(sig_ref) + len(sig_other) - 1
    n = int(2 ** np.ceil(np.log2(n)))  # prossima potenza di 2 per FFT veloce

    X = np.fft.rfft(sig_ref, n)
    Y = np.fft.rfft(sig_other, n)

    # PHAT weighting: normalizza per la magnitudine del prodotto incrociato
    # → riduce l'effetto di picchi spuri, più robusto in ambienti riverberanti
    G = X * np.conj(Y)
    denom = np.abs(G)
    denom[denom < 1e-10] = 1e-10  # evita divisione per zero
    G /= denom

    cc = np.fft.irfft(G, n)
    # Riorganizza in modo che il lag 0 sia al centro
    cc = np.fft.fftshift(cc)

    lag_samples = np.argmax(cc) - n // 2
    delay_sec = lag_samples / fs
    return delay_sec


def _estimate_position(delays):
    """
    Placeholder per la stima della posizione 3D dall'array microfonico.
    delays: array di (CHANNELS-1,) delay in secondi rispetto al canale REF.
    Sostituisci con il tuo algoritmo (es. least-squares su geometria nota).
    Restituisce un array [x, y, z] normalizzato in metri.
    """
    # Stima banale: mappa i primi due delay su x, y, z=0
    # Il segno del delay dà la direzione relativa al canale di riferimento.
    x = float(delays[0]) * 343.0  # velocità del suono 343 m/s
    y = float(delays[1]) * 343.0 if len(delays) > 1 else 0.0
    z = float(delays[2]) * 343.0 if len(delays) > 2 else 0.0
    return np.array([x, y, z], dtype=np.float32)


def run(audio_queue, result_queue, stop_event):
    """
    Entry point del thread processing.
    Loop: get blocco → accumula in buffer circolare → FFT + TDOA → position → result_queue.
    """

    # Buffer circolare per l'overlap-add
    # Shape: (FFT_WINDOW, CHANNELS) — mantiene gli ultimi FFT_WINDOW sample per canale
    circ_buffer = np.zeros((FFT_WINDOW, CHANNELS), dtype=np.float32)

    # Finestra di Hann applicata alla FFT per ridurre spectral leakage
    hann = np.hanning(FFT_WINDOW).astype(np.float32)

    logger.info("processing: avviato, FFT_WINDOW=%d, OVERLAP=%d", FFT_WINDOW, OVERLAP)

    while not stop_event.is_set():
        # --- Recupera il prossimo blocco dalla queue ---
        try:
            block, timestamp = audio_queue.get(timeout=0.1)
        except queue.Empty:
            # Timeout: nessun blocco disponibile, ricontrolla stop_event
            continue

        # --- Accumulo nel buffer circolare (shift + insert) ---
        # Shift a sinistra di BLOCKSIZE sample, inserisci il nuovo blocco in coda
        circ_buffer = np.roll(circ_buffer, -BLOCKSIZE, axis=0)
        circ_buffer[-BLOCKSIZE:, :] = block

        # --- FFT su tutta la finestra FFT_WINDOW con Hann ---
        # Shape windowed: (FFT_WINDOW, CHANNELS)
        windowed = circ_buffer * hann[:, np.newaxis]
        spectrum = np.fft.rfft(windowed, axis=0)
        # spectrum shape: (FFT_WINDOW//2 + 1, CHANNELS)
        # magnitudes in dB per la GUI
        magnitudes = 20 * np.log10(np.abs(spectrum) + 1e-10)

        # --- TDOA: GCC-PHAT tra canale REF e tutti gli altri ---
        delays = np.zeros(CHANNELS - 1, dtype=np.float32)
        ref_signal = circ_buffer[:, REF_CHANNEL]
        for ch in range(1, CHANNELS):
            delays[ch - 1] = _gcc_phat(ref_signal, circ_buffer[:, ch], SAMPLERATE)

        # --- Stima posizione 3D ---
        position = _estimate_position(delays)

        # --- Pacchetto risultati verso la GUI ---
        result = {
            "timestamp": timestamp,
            "position": position,  # np.array [x, y, z]
            "delays": delays,  # np.array (CHANNELS-1,) in secondi
            "magnitudes": magnitudes,  # np.array (FFT_WINDOW//2+1, CHANNELS) in dB
        }

        # put_nowait: se la GUI è indietro scartiamo il frame più vecchio
        try:
            result_queue.put_nowait(result)
        except queue.Full:
            try:
                result_queue.get_nowait()  # rimuove il frame più vecchio
                result_queue.put_nowait(result)
            except queue.Empty:
                pass

        audio_queue.task_done()

    logger.info("processing: terminato")
