from dataclasses import dataclass, field
from pathlib import Path
import logging
import time
import unittest

import numpy as np
from scipy.io import wavfile

# Import dei moduli di tracciamento reali
try:
    from rt_bat_tracker.tracking.common_functions import (
        calc_delay,
        calc_multich_delays,
    )
    from rt_bat_tracker.tracking.localisation_mpr2003 import (
        tristar_mellen_pachter,
    )
except ImportError:
    logger.error("Unable to import tracking modules.")


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ==========================================
# PERCORSI DELLA CARTELLA DATI
# ==========================================
CHUNK_LIB_PATH = Path(
    r"C:\Users\gioba\OneDrive\Documenti\GitHub\rt-bat-tracker\data\chunk_library"
)
MIC_LAYOUT_PATH = Path(
    r"C:\Users\gioba\OneDrive\Documenti\GitHub\rt-bat-tracker\data\mic_layout"
)


def load_mic_layout(filepath: Path) -> tuple[np.ndarray, np.ndarray]:
    """Carica un layout di microfoni da un file TXT/CSV e calcola il vettore normale

    del piano basato sui primi 3 microfoni.
    """
    # Caricamento array X,Y,Z (supporta delimitatori a virgola o spazi)
    delimiter = "," if "," in open(filepath).read(200) else None
    micxyz = np.loadtxt(filepath, delimiter=delimiter)
    print(f"{micxyz}, shape {micxyz.shape}")
    # Assicura la forma (N, 3)
    if micxyz.ndim == 1:
        micxyz = micxyz.reshape(-1, 3)

    # Calcolo del vettore normale dal piano formato dai primi 3 microfoni
    if len(micxyz) >= 3:
        v1 = micxyz[1] - micxyz[0]
        v2 = micxyz[2] - micxyz[0]
        normal = np.cross(v1, v2)
        normal_unit = normal / np.linalg.norm(normal)
    else:
        normal_unit = np.array([0.0, 0.0, 1.0])

    return micxyz, normal_unit


@dataclass
class Config:
    vsound: float = 343.0  # Velocità del suono in m/s


@dataclass
class LocAnalytics:
    """Dataclass per le metriche di analisi della localizzazione."""

    start_time: float = 0.0
    call_duration: float = 0.0
    max_rms: float = 0.0
    significant_channels: list = field(default_factory=list)
    processing_duration: float = 0.0
    chunk_size: int = 0
    cross_correlations: list = field(default_factory=list)
    channel_pairs: list = field(default_factory=list)
    TDOA: list = field(default_factory=list)
    TDOA_sum: float = 0.0


class ZeroSumLocalizer:

    def __init__(
        self,
        cfg: Config,
        significant_channels: list = None,
        max_speed: float = None,
    ):
        self.cfg = cfg
        self.significant_channels = (
            significant_channels
            if significant_channels is not None
            else [0, 1, 2, 3, 4]
        )
        self.max_speed = max_speed
        self.last_valid_loc = None
        self.last_call_time = None

    def process(
        self,
        chunk: np.ndarray,
        timestamp: float,
        micxyz: np.ndarray,
        normal_vector: np.ndarray,
        fs: int,
    ):
        """Elabora il chunk di audio SENZA dipendere da uno stato globale."""
        analytics = LocAnalytics(
            start_time=time.perf_counter_ns(),
            chunk_size=chunk.shape[0],
            significant_channels=self.significant_channels,
        )

        logger.debug(f"Processing call chunk with shape {chunk.shape}, fs={fs}")

        path_diff = []
        for ch1 in self.significant_channels:
            for ch2 in self.significant_channels:
                if ch1 != ch2:
                    two_ch = np.column_stack((chunk[:, ch1], chunk[:, ch2]))
                    tdoa, cc = calc_delay(two_ch, fs)

                    if ch1 == self.significant_channels[0]:
                        path_diff.append(tdoa * self.cfg.vsound)

                    analytics.cross_correlations.append(cc)
                    analytics.channel_pairs.append((ch1, ch2))
                    analytics.TDOA.append(tdoa)

        path_diff = np.array(path_diff)
        analytics.TDOA_sum = sum(analytics.TDOA)
        print(
            f"computing delays took {(time.perf_counter_ns() - analytics.start_time)*1e-6:.4f} ms, evaluated pairs: {analytics.channel_pairs}"
        )
        # Calcolo della posizione 3D
        locations = tristar_mellen_pachter(
            micxyz[self.significant_channels],
            path_diff,
            normal_vector,
        )

        # Gestione difensiva se la tristar restituisce []
        if locations is not None and len(locations) == 0:
            locations = None

        print(f"TDOA_sum: {analytics.TDOA_sum * 1e12:.4f} ps")

        # Durata del processing
        analytics.processing_duration = time.perf_counter_ns() - analytics.start_time
        print(f"Processing duration: {analytics.processing_duration / 1e6:.4f} ms")

        return locations, analytics


# ==========================================
# UNITTEST CON DATI DA DISCO (WAV + MIC)
# ==========================================


class TestLocalizerWithRealFiles(unittest.TestCase):

    def setUp(self):
        self.cfg = Config(vsound=343.0)

        # Carica il primo layout microfonico disponibile nella cartella mic_layout
        if not MIC_LAYOUT_PATH.exists():
            self.skipTest(f"Cartella mic_layout non trovata: {MIC_LAYOUT_PATH}")

        mic_files = list(MIC_LAYOUT_PATH.glob("adapted_triang.csv"))

        if not mic_files:
            self.skipTest(f"Nessun file mic_layout trovato in {MIC_LAYOUT_PATH}")

        self.layout_file = mic_files[0]
        self.micxyz, self.normal_vector = load_mic_layout(self.layout_file)
        print(
            f"\n[SETUP] Caricato mic layout da: {self.layout_file.name} (Forma: {self.micxyz.shape})"
        )

        self.localizer = ZeroSumLocalizer(
            cfg=self.cfg, significant_channels=[1, 2, 3, 4]
        )

    def test_process_all_chunks(self):
        """Esegue il test di localizzazione accoppiando il mic layout con tutti i file WAV."""
        if not CHUNK_LIB_PATH.exists():
            self.skipTest(f"Cartella non trovata: {CHUNK_LIB_PATH}")

        wav_files = list(CHUNK_LIB_PATH.glob("Event_141.wav"))
        if not wav_files:
            self.skipTest(f"Nessun file .wav trovato in {CHUNK_LIB_PATH}")

        print(f"[TEST] Trovati {len(wav_files)} file WAV per la localizzazione...")

        for wav_path in wav_files:
            print(f"\n---> Elaborazione WAV: {wav_path.name}")
            fs, audio_data = wavfile.read(wav_path)

            if audio_data.ndim < 2 or audio_data.shape[1] < 3:
                print(
                    f"Saltato {wav_path.name}: numero di canali insufficienti ({audio_data.shape})"
                )
                continue
            print(
                f"Audio shape: {audio_data.shape}, fs={fs}, mic layout shape: {self.micxyz.shape}"
            )
            locations, analytics = self.localizer.process(
                chunk=audio_data,
                timestamp=time.time(),
                micxyz=self.micxyz,
                normal_vector=self.normal_vector,
                fs=fs,
            )

            self.assertIsNotNone(analytics)
            self.assertGreater(analytics.processing_duration, 0)
            print(f"Posizioni calcolate: {locations}")


if __name__ == "__main__":
    unittest.main()
