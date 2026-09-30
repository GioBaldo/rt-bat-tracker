from dataclasses import dataclass, field
from pathlib import Path
import logging
import time
import unittest
from scipy import signal

import numpy as np
from scipy.io import wavfile

# Inizializzazione logger prima delle importazioni per evitare NameError
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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

    dal piano basato sui primi 3 microfoni.
    """
    # Lettura sicura del file con contesto 'with' per evitare ResourceWarning
    with open(filepath, "r", encoding="utf-8") as f:
        sample = f.read(200)
        delimiter = "," if "," in sample else None

    # Caricamento array X,Y,Z
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

    filter_order: int = 4
    cutoff_freq: float = 1000.0


@dataclass
class LocAnalytics:
    """Dataclass per le metriche di analisi della localizzazione."""

    start_time: float = 0.0
    call_duration: float = 0.0
    max_rms: float = 0.0
    significant_channels: list = field(default_factory=list)
    processing_duration: float = 0.0
    cc_duration: float = 0.0
    validation_duration: float = 0.0
    loc_duration: float = 0.0
    chunk_size: int = 0
    cross_correlations: list = field(default_factory=list)
    channel_pairs: list = field(default_factory=list)
    TDOA: list = field(default_factory=list)


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
            for ch2 in self.significant_channels[ch1 + 1 :]:

                two_ch = np.column_stack((chunk[:, ch1], chunk[:, ch2]))
                tdoa, cc = calc_delay(two_ch, fs)

                if ch1 == self.significant_channels[0]:
                    path_diff.append(tdoa * self.cfg.vsound)

                analytics.cross_correlations.append(cc)
                analytics.channel_pairs.append((ch1, ch2))
                analytics.TDOA.append(tdoa)
                print(
                    f"Pair ({ch1}, {ch2}): TDOA = {tdoa*1e6:.4f} µs, Path Diff = {tdoa*self.cfg.vsound:.4f} m - two_ch shape: {two_ch.shape} - CC length: {len(cc)}"
                )

        path_diff = np.array(path_diff)

        analytics.cc_duration = time.perf_counter_ns() - analytics.start_time
        print(
            f"computing cross correlations took {analytics.cc_duration / 1e6:.4f} ms, pairs: {analytics.channel_pairs}"
        )
        selected_mics = self.velidate_tdoa(
            analytics.TDOA, analytics.channel_pairs, self.significant_channels, keep=5
        )

        analytics.validation_duration = (
            time.perf_counter_ns() - analytics.start_time - analytics.cc_duration
        )
        print(f"validation duration: {analytics.validation_duration / 1e6:.4f} ms")

        # Calcolo della posizione 3D
        locations = tristar_mellen_pachter(
            micxyz[selected_mics],
            path_diff[selected_mics[1:]],
            normal_vector,
        )

        # Gestione difensiva se la tristar restituisce []
        if locations is not None and len(locations) == 0:
            locations = None

        analytics.loc_duration = (
            time.perf_counter_ns()
            - analytics.validation_duration
            - analytics.cc_duration
            - analytics.start_time
        )
        print(f"Localisation duration: {analytics.loc_duration / 1e6:.4f} ms")

        # Durata del processing
        analytics.processing_duration = time.perf_counter_ns() - analytics.start_time
        print(f"Processing duration: {analytics.processing_duration / 1e6:.4f} ms")

        return locations, analytics

    def velidate_tdoa(self, tdoa_vec, pairs_vec, channels, keep=6):
        """comparest tdoas and checks zero sum condition between pairs to validate each microphone's measurements"""
        n_channels = len(channels)
        mic_errors = np.zeros(n_channels)

        for idx1 in range(n_channels):
            ch1 = channels[idx1]
            for idx2 in range(idx1 + 1, n_channels):
                ch2 = channels[idx2]
                pair12 = (ch1, ch2)
                try:
                    tdoa12 = tdoa_vec[pairs_vec.index(pair12)]
                except ValueError:
                    print(f"Pair {pair12} not found in pairs_vec. Skipping.")
                    continue

                for idx3 in range(idx2 + 1, n_channels):
                    ch3 = channels[idx3]
                    pair13 = (ch1, ch3)
                    pair23 = (ch2, ch3)

                    try:
                        tdoa13 = tdoa_vec[pairs_vec.index(pair13)]
                        tdoa23 = tdoa_vec[pairs_vec.index(pair23)]

                    except ValueError:
                        print(
                            f"Error occurred while fetching TDOA values: {pair13} or {pair23} not found in pairs_vec. Skipping. idx1={idx1}, idx2={idx2}, idx3={idx3}, channels={channels}"
                        )
                        continue

                    error = abs(tdoa12 + tdoa23 - tdoa13)
                    mic_errors[idx1] += error
                    mic_errors[idx2] += error
                    mic_errors[idx3] += error
                    # print(f"Error for triplet ({ch1}, {ch2}, {ch3}): {error:.6f} s")

        best_idx = np.argsort(mic_errors.flatten())[:keep]
        selected_mics = np.unique(
            np.array(np.unravel_index(best_idx, mic_errors.shape)).flatten()
        )

        print(f"Selected microphones based on zero-sum validation: {selected_mics}")
        print(f"Mic errors: {mic_errors}")

        return selected_mics

    def plot_analytics(self, analytics: LocAnalytics, chunk: np.ndarray = None):
        """Plot delle metriche di analisi e dei canali audio separati.

        Layout (5 colonne):
        - Riga 1: 5 subplot separati (uno per canale audio).
        - Righe 2+: Subplot delle cross-correlazioni organizzati su 5 colonne.
        """
        import matplotlib.pyplot as plt

        num_pairs = len(analytics.cross_correlations)
        cols = 5

        # Calcolo delle righe necessarie: 1 riga per i 5 canali audio + righe per le CC
        cc_rows = (
            (num_pairs + cols - 1) // cols if num_pairs > 0 else 0
        )  # Esempio: 20 / 5 = 4 righe
        total_rows = 1 + cc_rows  # Totale righe (1 audio + CC)

        fig = plt.figure(figsize=(18, 2.5 * total_rows), constrained_layout=True)
        fig.suptitle(
            "Localisation Analytics - Audio Channels & Cross-Correlations",
            fontsize=15,
            fontweight="bold",
        )

        gs = fig.add_gridspec(total_rows, cols)

        # ----------------------------------------------------
        # RIGA 1: 5 Subplot separati per i canali audio
        # ----------------------------------------------------
        if chunk is not None:
            num_samples, num_channels = chunk.shape
            time_axis = np.arange(num_samples)
            colors_audio = plt.cm.Set1(np.linspace(0, 1, max(num_channels, 5)))

            for ch in range(min(num_channels, cols)):
                ax_ch = fig.add_subplot(gs[0, ch])
                ax_ch.plot(
                    time_axis,
                    chunk[:, ch],
                    color=colors_audio[ch],
                    linewidth=0.8,
                )
                ax_ch.set_title(
                    f"Channel {ch}",
                    fontsize=10,
                    fontweight="semibold",
                    color="navy",
                )
                ax_ch.grid(True, linestyle=":", alpha=0.6)
                ax_ch.tick_params(axis="both", labelsize=8)

                if ch == 0:
                    ax_ch.set_ylabel("Amplitude", fontsize=8)
                ax_ch.set_xlabel("Samples", fontsize=8)

        # ----------------------------------------------------
        # RIGHE 2+: Grid per le Cross-Correlazioni (5 colonne)
        # ----------------------------------------------------
        if num_pairs > 0:
            cc_colors = plt.cm.viridis(np.linspace(0.1, 0.9, num_pairs))

            for i, cc in enumerate(analytics.cross_correlations):
                # Posizionamento in griglia a partire dalla riga index 1
                r = 1 + (i // cols)
                c = i % cols

                ax_cc = fig.add_subplot(gs[r, c])
                pair = (
                    analytics.channel_pairs[i]
                    if i < len(analytics.channel_pairs)
                    else (i,)
                )

                # Traiettoria Cross-Correlazione
                ax_cc.plot(cc, color=cc_colors[i], linewidth=1.1)

                # Marker sul picco massimo (Lag stimato)
                max_idx = np.argmax(cc)
                max_val = cc[max_idx]
                ax_cc.plot(max_idx, max_val, "ro", markersize=3.5)
                ax_cc.axvline(
                    x=max_idx,
                    color="red",
                    linestyle="--",
                    alpha=0.5,
                    linewidth=0.8,
                )

                # Styling del subplot
                ax_cc.set_title(f"Pair {pair}", fontsize=9, fontweight="semibold")
                ax_cc.grid(True, linestyle=":", alpha=0.6)
                ax_cc.tick_params(axis="both", labelsize=8)

                # Etichetta TDOA integrata
                if i < len(analytics.TDOA):
                    tdoa_us = analytics.TDOA[i] * 1e6
                    ax_cc.text(
                        0.03,
                        0.80,
                        f"TDOA: {tdoa_us:.1f} µs",
                        transform=ax_cc.transAxes,
                        fontsize=7,
                        bbox=dict(
                            boxstyle="round,pad=0.2",
                            facecolor="white",
                            alpha=0.75,
                            edgecolor="none",
                        ),
                    )

                if c == 0:
                    ax_cc.set_ylabel("Correlation", fontsize=8)
                if r == total_rows - 1:
                    ax_cc.set_xlabel("Lag", fontsize=8)

        plt.show()

    def _highpass_filter(self, block):
        """
        Apply a highpass Butterworth filter to the block.
        Returns the filtered block of the same shape.
        """
        b, a = signal.butter(
            1,
            20000 / (192000 * 0.5),
            "high",
        )
        return signal.lfilter(b, a, block, axis=0)


# ==========================================
# UNITTEST CON DATI DA DISCO (WAV + MIC)
# ==========================================


class TestLocalizerWithRealFiles(unittest.TestCase):

    def setUp(self):
        self.cfg = Config(vsound=340.0)

        # Carica il primo layout microfonico disponibile nella cartella mic_layout
        if not MIC_LAYOUT_PATH.exists():
            self.skipTest(f"Cartella mic_layout non trovata: {MIC_LAYOUT_PATH}")

        mic_files = list(MIC_LAYOUT_PATH.glob("single_bat_1234.csv"))

        if not mic_files:
            self.skipTest(f"Nessun file mic_layout trovato in {MIC_LAYOUT_PATH}")

        self.layout_file = mic_files[0]
        self.micxyz, self.normal_vector = load_mic_layout(self.layout_file)
        print(
            f"\n[SETUP] Caricato mic layout da: {self.layout_file.name} (Forma: {self.micxyz.shape})"
        )

        self.localizer = ZeroSumLocalizer(
            cfg=self.cfg, significant_channels=[0, 1, 2, 3, 4, 5, 6, 7]
        )

    def test_process_all_chunks(self):
        """Esegue il test di localizzazione accoppiando il mic layout con tutti i file WAV."""
        if not CHUNK_LIB_PATH.exists():
            self.skipTest(f"Cartella non trovata: {CHUNK_LIB_PATH}")

        wav_files = list(CHUNK_LIB_PATH.glob("single_bat_1234_01.wav"))
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
            audio_data = self.localizer._highpass_filter(audio_data)

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
            self.localizer.plot_analytics(analytics, audio_data)


if __name__ == "__main__":
    unittest.main()
