from dataclasses import dataclass, field
from pathlib import Path
import logging
import time
import unittest
from scipy import signal
import numpy as np
from scipy.io import wavfile

# Inizializzazione logger prima delle importazioni per evitare NameError
# logging.basicConfig(
#     level=logging.DEBUG, format="%(name)s - %(levelname)s - %(message)s"
# )
logging.getLogger("matplotlib").setLevel(logging.WARNING)

logger = logging.getLogger("LOC")
logger.setLevel(logging.INFO)

# Import dei moduli di tracciamento reali
try:
    from rt_bat_tracker.tracking.common_functions import (
        calc_delay,
        calc_multich_delays,
    )
    from rt_bat_tracker.tracking.localisation_mpr2003 import (
        tristar_mellen_pachter,
    )
    from rt_bat_tracker.utils.analytics_plots import plot_cross_correlations
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
    logger.debug(f"{micxyz}, shape {micxyz.shape}")

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
    vsound: float = 340.0  # Velocità del suono in m/s

    filter_order: int = 4
    cutoff_freq: float = 1000.0


@dataclass
class LocAnalytics:

    start_time: float = 0.0
    call_duration: float = 0.0
    max_rms: float = 0.0
    significant_channels: list = field(default_factory=list)
    processing_duration: float = 0.0
    cc_duration: float = 0.0
    validation_duration: float = 0.0
    loc_duration: float = 0.0
    chunk_size: int = 0
    selected_mics: list = field(default_factory=list)
    selected_mics_str: str = ""
    cross_correlations: list = field(default_factory=list)
    channel_pairs: list = field(default_factory=list)
    TDOA: list = field(default_factory=list)
    TDOA_sum: float = 0.0
    cc_errors: list = field(default_factory=list)
    mic_errors: list = field(default_factory=list)


class ZeroSumLocalizer:

    def __init__(
        self,
        cfg: Config,
        significant_channels: list = None,
        max_speed: float = None,
    ):
        self.cfg = cfg
        self.max_speed = max_speed
        self.last_valid_loc = None
        self.last_call_time = None

        self.cc_errors = None

    def get_zerosum_delays(
        self,
        chunk: np.ndarray,
        fs: int,
        ch_to_keep: int = 4,
        significant_channels: list = [0, 1, 2, 3, 4],
    ):
        """compute cross-correlation between all unique copules of mics and check zero-sum residual between triplets.
        Returns the set of delays between the best rated sensors

        Args:
            chunk (np.ndarray): n channel audio chunk (n_samples, n_channels)
            fs (int): sampling frequency
            ch_to_keep (int, optional): number of channels to keep. Defaults to 4 (minimum for 3D localisation).

        Returns:
            path_diff (np.ndarray): array of path differences for the selected channels
            selected_mics (np.ndarray): indices of the selected microphones
            analytics (LocAnalytics): object containing analysis metrics
        """

        analytics = LocAnalytics(
            chunk_size=chunk.shape[0],
            significant_channels=significant_channels,
        )

        logger.debug(
            f"Processing call chunk with shape {chunk.shape}, fs={fs}, significant_channels={significant_channels}"
        )

        n_channels = len(significant_channels)

        for idx1 in range(n_channels):
            ch1 = significant_channels[idx1]
            for idx2 in range(idx1 + 1, n_channels):
                ch2 = significant_channels[idx2]

                two_ch = np.column_stack((chunk[:, ch1], chunk[:, ch2]))
                tdoa, cc = calc_delay(two_ch, fs)

                analytics.cross_correlations.append(cc)
                analytics.channel_pairs.append((ch1, ch2))
                analytics.TDOA.append(tdoa)
                logger.debug(
                    f"Pair ({ch1}, {ch2}): TDOA = {tdoa*1e6:.4f} µs, Path Diff = {tdoa*self.cfg.vsound:.4f} m - two_ch shape: {two_ch.shape} - CC length: {len(cc)}"
                )

        analytics.cc_duration = time.perf_counter_ns() - analytics.start_time
        logger.debug(f"pairs: {analytics.channel_pairs}")
        (
            selected_mics,
            path_diff,
            analytics.cc_errors,
            analytics.mic_errors,
            analytics.TDOA_sum,
        ) = self.velidate_tdoa(
            analytics.TDOA,
            analytics.channel_pairs,
            significant_channels,
            keep=ch_to_keep,
        )

        analytics.selected_mics = selected_mics
        analytics.selected_mics_str = "".join(map(str, selected_mics))

        analytics.validation_duration = (
            time.perf_counter_ns() - analytics.start_time - analytics.cc_duration
        )

        return path_diff, selected_mics, analytics

    def velidate_tdoa(self, tdoa_vec, pairs_vec, channels, keep=6):
        """comparest tdoas and checks zero sum condition between pairs to validate each microphone's measurements"""
        n_channels = len(channels)
        mic_errors = np.zeros(n_channels)
        cc_errors = np.zeros(len(pairs_vec))

        if keep > n_channels:
            keep = n_channels

        for idx1 in range(n_channels):
            ch1 = channels[idx1]
            for idx2 in range(idx1 + 1, n_channels):
                ch2 = channels[idx2]
                pair12 = (ch1, ch2)
                try:
                    idx12 = pairs_vec.index(pair12)
                    tdoa12 = tdoa_vec[idx12]
                except ValueError:
                    logger.warning(f"Pair12 {pair12} not found in pairs_vec. Skipping.")
                    continue

                for idx3 in range(idx2 + 1, n_channels):
                    ch3 = channels[idx3]
                    pair13 = (ch1, ch3)
                    pair23 = (ch2, ch3)

                    try:
                        idx13 = pairs_vec.index(pair13)
                        idx23 = pairs_vec.index(pair23)
                        tdoa13 = tdoa_vec[idx13]
                        tdoa23 = tdoa_vec[idx23]

                    except ValueError:
                        logger.error(
                            f"Error occurred while fetching TDOA values: {pair13} or {pair23} not found in pairs_vec. Skipping. idx1={idx1}, idx2={idx2}, idx3={idx3}, channels={channels}"
                        )
                        continue

                    error = abs(tdoa12 + tdoa23 - tdoa13)
                    mic_errors[idx1] += error
                    mic_errors[idx2] += error
                    mic_errors[idx3] += error
                    cc_errors[idx12] += error
                    cc_errors[idx23] += error
                    cc_errors[idx13] += error

        best_idx = np.argsort(mic_errors.flatten())[:keep]
        # selected_mics = np.unique(
        #     np.array(np.unravel_index(best_idx, mic_errors.shape)).flatten()
        # )
        selected_mics = np.sort(np.array(channels)[best_idx])

        logger.debug(f"Selected microphones after zero-sum validation: {selected_mics}")

        path_diff = []

        TDOA_sum = 0.0
        for idx1 in range(len(selected_mics) - 1):
            ch1 = selected_mics[idx1]
            ch2 = selected_mics[idx1 + 1]
            pair = (ch1, ch2)

            abs_pair = (selected_mics[0], ch2)  # Pair from first to current mic

            if abs_pair in pairs_vec:
                abs_TDOA = tdoa_vec[pairs_vec.index(abs_pair)]
                path_diff.append(abs_TDOA * self.cfg.vsound)
            else:
                logger.warning(
                    f"Pair ({abs_pair[0]},{abs_pair[1]}) not found in pairs_vec during TDOA sum calculation."
                )

            if pair in pairs_vec:
                TDOA = tdoa_vec[pairs_vec.index(pair)]
                TDOA_sum += TDOA
            else:
                logger.warning(
                    f"Pair ({ch1},{ch2}) not found in pairs_vec during TDOA sum calculation."
                )

        loop_pair = (selected_mics[0], selected_mics[-1])
        if loop_pair in pairs_vec:
            logger.debug(
                f"Subtracting TDOA for pair ({loop_pair[0]},{loop_pair[1]}): {tdoa_vec[pairs_vec.index(loop_pair)]*1e6:.6f} μs"
            )
            TDOA_sum -= tdoa_vec[pairs_vec.index(loop_pair)]
        else:
            logger.warning(
                f"Pair {loop_pair} not found in pairs_vec during TDOA sum calculation."
            )

        logger.info(
            f"Selected microphones based on zero-sum validation: {selected_mics}, LOOP TDOA sum: {TDOA_sum*1e6:.6f} μs"
        )
        logger.debug(f"Mic errors: {mic_errors}")

        return (
            selected_mics,
            np.array(path_diff),
            cc_errors.tolist(),
            mic_errors.tolist(),
            TDOA_sum,
        )

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

        mic_files = list(MIC_LAYOUT_PATH.glob("single_bat_1234.csv"))  # <------

        if not mic_files:
            self.skipTest(f"Nessun file mic_layout trovato in {MIC_LAYOUT_PATH}")

        self.layout_file = mic_files[0]
        self.micxyz, self.normal_vector = load_mic_layout(self.layout_file)
        logger.debug(
            f"\n[SETUP] Caricato mic layout da: {self.layout_file.name} (Forma: {self.micxyz.shape})"
        )

        self.localizer = ZeroSumLocalizer(
            cfg=self.cfg, significant_channels=[0, 1, 2, 3, 4, 5, 6, 7]  # <------
        )

    def test_process_all_chunks(self):
        """Esegue il test di localizzazione accoppiando il mic layout con tutti i file WAV."""
        if not CHUNK_LIB_PATH.exists():
            self.skipTest(f"Cartella non trovata: {CHUNK_LIB_PATH}")

        wav_files = list(CHUNK_LIB_PATH.glob("single_bat_1234_00.wav"))  # <------
        if not wav_files:
            self.skipTest(f"Nessun file .wav trovato in {CHUNK_LIB_PATH}")

        logger.debug(
            f"[TEST] Trovati {len(wav_files)} file WAV per la localizzazione..."
        )

        for wav_path in wav_files:

            fs, audio_data = wavfile.read(wav_path)

            if audio_data.ndim < 2 or audio_data.shape[1] < 3:
                logger.info(
                    f"Saltato {wav_path.name}: numero di canali insufficienti ({audio_data.shape})"
                )
                continue
            logger.debug(
                f"Audio shape: {audio_data.shape}, fs={fs}, mic layout shape: {self.micxyz.shape}"
            )
            audio_data = self.localizer._highpass_filter(audio_data)

            path_diff, selected_mics, analytics = self.localizer.get_zerosum_delays(
                chunk=audio_data,
                fs=fs,
                ch_to_keep=5,
                significant_channels=[0, 1, 2, 3, 4, 5, 6, 7],
            )

            logger.debug(
                f"Passing {self.micxyz[selected_mics]} to tristar_mellen_pachter with path_diff {path_diff} for selected mics {selected_mics} and normal vector {self.normal_vector}"
            )

            # Calcolo della posizione 3D
            locations = tristar_mellen_pachter(
                self.micxyz[selected_mics],
                path_diff,
                [-1, -1, 0],  # self.normal_vector,
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
            logger.info(
                f"cross correlation duration: {analytics.cc_duration / 1e6:.4f} ms,"
            )
            logger.info(
                f"validation duration: {analytics.validation_duration / 1e6:.4f} ms,"
            )
            logger.info(f"localisation duration: {analytics.loc_duration / 1e6:.4f} ms")

            # Durata del processing
            analytics.processing_duration = (
                time.perf_counter_ns() - analytics.start_time
            )
            logger.info(
                f"total processing duration: {analytics.processing_duration / 1e6:.4f} ms"
            )

            self.assertIsNotNone(analytics)
            self.assertGreater(analytics.processing_duration, 0)

            logger.info(
                f"computed locations: {locations} - TDOA sum: {analytics.TDOA_sum*1e6:.6f} us"
            )

            import matplotlib.pyplot as plt

            plot_cross_correlations(analytics, audio_data, fig=plt.figure())
            plt.show()


if __name__ == "__main__":
    unittest.main()
