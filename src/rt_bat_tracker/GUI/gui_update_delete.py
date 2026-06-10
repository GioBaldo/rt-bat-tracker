"""
gui_update.py
GUI con pyqtgraph + OpenGL.
Il thread di processing NON tocca mai i widget Qt direttamente.
Usa un Qt Signal per passare i dati al main thread Qt in modo thread-safe.
"""

import logging
import queue
import time

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt5.QtCore import QObject, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
)

logger = logging.getLogger(__name__)

TARGET_FPS = 20
FRAME_MS = int(1000 / TARGET_FPS)  # 50 ms per frame


# ---------------------------------------------------------------------------
# Worker QObject: vive nel main thread Qt, riceve i dati dalla result_queue
# tramite un QTimer e li emette come Signal — nessun lock manuale necessario.
# ---------------------------------------------------------------------------
class DataPoller(QObject):
    """
    Fa polling della result_queue con un QTimer a 20 fps.
    Emette new_data ogni volta che c'è un risultato disponibile.
    Questo è il bridge thread-safe tra il thread processing e Qt.
    """

    new_data = pyqtSignal(dict)

    def __init__(self, result_queue, parent=None):
        super().__init__(parent)
        self._queue = result_queue
        self._timer = QTimer(self)
        self._timer.setInterval(FRAME_MS)
        self._timer.timeout.connect(self._poll)

    def start(self):
        self._timer.start()

    def stop(self):
        self._timer.stop()

    def _poll(self):
        """Chiamato ogni FRAME_MS ms nel main thread Qt."""
        try:
            result = self._queue.get_nowait()
            self.new_data.emit(result)
        except queue.Empty:
            pass  # nessun dato nuovo: non aggiornare la GUI, risparmia CPU


# ---------------------------------------------------------------------------
# Finestra principale
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self, result_queue):
        super().__init__()
        self.setWindowTitle("Array microfonico — posizione 3D")
        self.resize(1200, 700)

        # --- Layout ---
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # Pannello sinistro: vista 3D OpenGL
        self._setup_3d_view(root)

        # Pannello destro: spettro FFT + info numeriche
        right_panel = QVBoxLayout()
        root.addLayout(right_panel, stretch=1)
        self._setup_spectrum(right_panel)
        self._setup_info_labels(right_panel)

        # --- Poller dati ---
        self._poller = DataPoller(result_queue, parent=self)
        self._poller.new_data.connect(self._on_new_data)
        self._poller.start()

        logger.info("GUI: finestra creata")

    # ------------------------------------------------------------------
    # Setup widget
    # ------------------------------------------------------------------

    def _setup_3d_view(self, parent_layout):
        """Vista OpenGL 3D per la posizione stimata."""
        self._gl_view = gl.GLViewWidget()
        self._gl_view.setMinimumWidth(600)
        self._gl_view.setCameraPosition(distance=3.0)

        # Griglia di riferimento
        grid = gl.GLGridItem()
        grid.setSize(4, 4)
        grid.setSpacing(0.5, 0.5)
        self._gl_view.addItem(grid)

        # Assi XYZ
        for vec, color in [
            ([1, 0, 0], (1, 0, 0, 1)),
            ([0, 1, 0], (0, 1, 0, 1)),
            ([0, 0, 1], (0, 0, 1, 1)),
        ]:
            axis = gl.GLLinePlotItem(
                pos=np.array([[0, 0, 0], vec], dtype=np.float32),
                color=color,
                width=1.5,
                antialias=True,
            )
            self._gl_view.addItem(axis)

        # Punto sorgente stimata posizione
        self._source_point = gl.GLScatterPlotItem(
            pos=np.array([[0, 0, 0]], dtype=np.float32),
            color=(1, 0.5, 0, 1),
            size=12,
            pxMode=True,
        )
        self._gl_view.addItem(self._source_point)

        # Traccia storia posizioni (ultimi N punti)
        self._pos_history = np.zeros((60, 3), dtype=np.float32)
        self._pos_trail = gl.GLLinePlotItem(
            pos=self._pos_history, color=(1, 0.5, 0, 0.4), width=1.0, antialias=True
        )
        self._gl_view.addItem(self._pos_trail)

        parent_layout.addWidget(self._gl_view, stretch=2)

    def _setup_spectrum(self, parent_layout):
        """Plot spettro FFT del canale di riferimento."""
        self._spectrum_plot = pg.PlotWidget(title="Spettro FFT — canale 0")
        self._spectrum_plot.setLabel("left", "Ampiezza", units="dB")
        self._spectrum_plot.setLabel("bottom", "Frequenza", units="Hz")
        self._spectrum_plot.setYRange(-80, 0)
        self._spectrum_plot.setXRange(0, 96_000)  # 0 → Nyquist a 192kHz
        self._spectrum_plot.showGrid(x=True, y=True, alpha=0.3)
        self._spectrum_curve = self._spectrum_plot.plot(
            pen=pg.mkPen(color=(100, 200, 255), width=1)
        )
        parent_layout.addWidget(self._spectrum_plot, stretch=2)

    def _setup_info_labels(self, parent_layout):
        """Label numeriche per posizione e delay."""
        self._lbl_pos = QLabel("Posizione:  —")
        self._lbl_delays = QLabel("Delay (ms): —")
        self._lbl_fps = QLabel("GUI fps:    —")
        for lbl in (self._lbl_pos, self._lbl_delays, self._lbl_fps):
            lbl.setStyleSheet("font-family: monospace; font-size: 12px;")
            parent_layout.addWidget(lbl)

        self._last_frame_time = time.monotonic()

    # ------------------------------------------------------------------
    # Aggiornamento GUI — chiamato nel main thread Qt via Signal
    # ------------------------------------------------------------------

    def _on_new_data(self, result):
        """
        Riceve il dict dal DataPoller (già nel main thread Qt).
        Aggiorna tutti i widget in un'unica passata.
        """
        pos = result["position"]  # np.array [x, y, z]
        delays = result["delays"]  # np.array (CHANNELS-1,)
        mags = result["magnitudes"]  # np.array (FFT_BIN, CHANNELS)

        # --- 3D: aggiorna punto e traccia ---
        self._source_point.setData(pos=pos[np.newaxis, :])

        self._pos_history = np.roll(self._pos_history, -1, axis=0)
        self._pos_history[-1] = pos
        self._pos_trail.setData(pos=self._pos_history)

        # --- Spettro: solo canale 0 ---
        n_bins = mags.shape[0]
        freqs = np.linspace(0, 96_000, n_bins)
        self._spectrum_curve.setData(freqs, mags[:, 0])

        # --- Label numeriche ---
        self._lbl_pos.setText(
            f"Posizione:  x={pos[0]:+.3f} m  y={pos[1]:+.3f} m  z={pos[2]:+.3f} m"
        )
        delay_ms = delays * 1000
        self._lbl_delays.setText(
            "Delay (ms): "
            + "  ".join(f"ch{i+1}={d:+.3f}" for i, d in enumerate(delay_ms))
        )

        # FPS effettivo GUI
        now = time.monotonic()
        fps = 1.0 / max(now - self._last_frame_time, 1e-6)
        self._last_frame_time = now
        self._lbl_fps.setText(f"GUI fps:    {fps:.1f}")

    def closeEvent(self, event):
        """Ferma il poller quando la finestra viene chiusa."""
        self._poller.stop()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Entry point del thread GUI
# ---------------------------------------------------------------------------


def run(result_queue, stop_event, app=None):
    """
    Deve essere chiamato dal main thread (Qt lo richiede su molte piattaforme).
    Su RPi4 con eglfs il QApplication va creato qui o nel main prima di chiamare run().
    """
    if app is None:
        import sys

        app = QApplication.instance() or QApplication(sys.argv)

    window = MainWindow(result_queue)
    window.show()

    # Quando la finestra viene chiusa, setta stop_event per fermare tutti i thread
    app.aboutToQuit.connect(stop_event.set)

    app.exec_()

    logger.info("gui_update: terminato")
