"""
audio_input.py
Gestisce l'acquisizione audio via sounddevice.
Il callback PortAudio è il metronomo hardware: non fare mai
operazioni lente qui dentro. Solo threshold check + put_nowait.
"""

import numpy as np
import sounddevice as sd
import logging

logger = logging.getLogger(__name__)

SAMPLERATE = 192_000
CHANNELS = 8
BLOCKSIZE = 1024  # sample per blocco
DTYPE = "float32"  # PortAudio converte internamente
THRESHOLD = 0.01  # RMS minimo per inviare il blocco al processing


def run(audio_queue, stop_event):
    """
    Entry point del thread audio.
    Apre lo stream e resta in attesa finché stop_event non viene settato.
    """

    def _callback(indata, frames, time_info, status):
        """
        Chiamato da PortAudio ogni BLOCKSIZE sample.
        Gira in un thread interno C di PortAudio — priorità real-time.
        NON usare logging, print o lock pesanti qui dentro.
        """
        if status:
            # Overflow/underflow: logga in modo asincrono tramite la queue
            # (non usare logger.warning direttamente — può bloccare)
            pass

        # Threshold check veloce: RMS su tutti i canali
        # indata shape: (BLOCKSIZE, CHANNELS), dtype float32
        rms = np.sqrt(np.mean(indata**2))
        if rms < THRESHOLD:
            return  # silenzio: scarta il blocco

        # Copia OBBLIGATORIA: il buffer indata viene riusato da PortAudio
        # subito dopo il return. Senza copy i dati nel processing sono corrotti.
        block = indata.copy()

        # Timestamp hardware ADC: secondi assoluti dalla scheda audio.
        # Fondamentale per il TDOA — è il riferimento temporale preciso.
        timestamp = time_info.inputBufferAdcTime

        # put_nowait: non blocca mai il callback.
        # Se la queue è piena (processing lento) scartiamo il blocco —
        # meglio perdere un frame che introdurre latenza accumulata.
        try:
            audio_queue.put_nowait((block, timestamp))
        except Exception:
            pass  # queue piena: frame scartato, va bene

    # Apertura stream
    try:
        with sd.InputStream(
            samplerate=SAMPLERATE,
            channels=CHANNELS,
            blocksize=BLOCKSIZE,
            dtype=DTYPE,
            callback=_callback,
            latency="low",  # chiede la latenza minima al driver
        ):
            logger.info(
                "Stream audio aperto: %d Hz, %d canali, blocksize=%d",
                SAMPLERATE,
                CHANNELS,
                BLOCKSIZE,
            )
            # Blocca il thread finché stop_event non viene settato dal main
            stop_event.wait()

    except sd.PortAudioError as e:
        logger.error("Errore PortAudio: %s", e)
        stop_event.set()  # propaga l'errore a tutti i thread

    logger.info("audio_input: terminato")
