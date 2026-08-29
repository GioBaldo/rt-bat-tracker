import logging
import sys
import time
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.uic import loadUi

logger = logging.getLogger("GUIUPDATE")
logger.setLevel(logging.DEBUG)


def run(state, cfg, session):
    QApplication.setAttribute(Qt.AA_UseDesktopOpenGL)
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)

    app = QApplication(sys.argv)

    window = MainWindow(state, cfg, session)
    window.show()

    app.aboutToQuit.connect(window.stop)

    app.exec_()
    state.gui_running_flag = False


class MainWindow(QMainWindow):

    def __init__(self, state, cfg, session):
        super().__init__()

        try:
            # Passiamo 'self' come secondo argomento per iniettare i widget come attributi della classe
            loadUi(
                cfg.GUIpath,
                self,
                {
                    "GLViewWidget": gl.GLViewWidget,
                    "PlotWidget": pg.PlotWidget,
                },
            )

        except Exception as e:
            logger.error(
                "unable to open the GUI Layout: %s \n%s ", cfg.GUIpath, e
            )
            state.stop(__name__)
            return

        self._session = session
        self._state = state
        self._cfg = cfg
        self.timer = int(1000 / self._cfg.update_fps)
        self.setWindowTitle(self._session.session_name)

        # import widgets and connect signals
        self._import_ui_widgets()
        self._connect_ui_signals()

        # Inizializzazione dei visualizzatori
        try:
            self._setup_path_viewer(self._state.micxyz)
            self._setup_spec_viewer()
            self._setup_vu_meter()
        except Exception as e:
            logger.error(
                "unable to initialize pathViewer or secondary widgets: %s", e
            )
            state.stop(__name__)

        self._poller = QTimer(self)
        self._poller.setInterval(self.timer)
        self._poller.timeout.connect(self._update)
        self._poller.start()

    def _connect_signal_safely(self, widget_name, signal_name, slot):
        """Metodo helper per connettere segnali evitando crash se il widget non esiste."""
        widget = getattr(self, widget_name, None)
        if widget is not None:
            signal = getattr(widget, signal_name, None)
            if signal is not None:
                signal.connect(slot)
            else:
                logger.warning(
                    f"Il segnale '{signal_name}' non esiste sul widget '{widget_name}'."
                )
        else:
            logger.error(
                f"Widget '{widget_name}' non trovato nell'interfaccia UI! Controlla l'objectName su Qt Designer."
            )

    def _connect_ui_signals(self):
        """Collega i segnali usando il wrapper sicuro."""
        self._connect_signal_safely("ExitButton", "clicked", self.stop)
        self._connect_signal_safely(
            "PlayButton", "clicked", self._on_play_pause_clicked
        )
        self._connect_signal_safely(
            "LeftButton", "clicked", self._on_slowmo_clicked
        )
        self._connect_signal_safely(
            "RightButton", "clicked", self._on_stop_clicked
        )
        self._connect_signal_safely(
            "pushButton_4", "clicked", self._on_mode_toggle_clicked
        )
        self._connect_signal_safely(
            "ThresholdSlider", "valueChanged", self._on_threshold_changed
        )
    def _import_ui_widgets(self):
        """Importa i widget dalla UI e li assegna come attributi della classe."""
        self.ExitButton = getattr(self, "ExitButton", None)
        self.PlayButton = getattr(self, "PlayButton", None)
        self.LeftButton = getattr(self, "LeftButton", None)
        self.RightButton = getattr(self, "RightButton", None)
        self.ToggleButton = getattr(self, "pushButton_4", None)


    def _update(self):
        if self._state.stop_event.isSet():
            self._session.update(None, None)
            self.stop()
            return
        if not self._state.gui_running_flag:
            logger.info(
                "GUI LOADED: time elapsed for loading - %4f",
                time.time() - self._state.gui_t_start,
            )
            self._state.gui_running_flag = True

        self._session.write_audiofile()
        self._update_path_viewer()
        self._update_spec_viewer()
        self._update_vu_meter()

##VIEWERS FUNCTIONS##
##PATH VIEWER###
    def _setup_path_viewer(self, micxyz):
        path_viewer = getattr(self, "pathViewer", None)
        if path_viewer is None:
            logger.warning("Widget 'pathViewer' non trovato nella UI.")
            return

        grid = gl.GLGridItem()
        grid.setSize(20, 20)
        grid.setSpacing(0.5, 0.5)
        path_viewer.addItem(grid)

        self._mic_plot = gl.GLScatterPlotItem(
            pos=micxyz,
            color=(1, 0, 0, 1),
            size=10,
        )
        path_viewer.addItem(self._mic_plot)

        self._source_plot = gl.GLScatterPlotItem(
            pos=np.array([[0, 0, 0]], dtype=np.float64),
            color=(1, 1, 0, 1),
            size=10,
        )
        path_viewer.addItem(self._source_plot)

        path_viewer.setCameraPosition(distance=20, azimuth=-50, elevation=30)

    def _update_path_viewer(self):
        pos, timestamp = self._state.get_result()
        logger.debug(
            f"GUI received this point: {pos} [timestamp: {timestamp}]"
        )
        self._session.update(pos, timestamp)
        points, colors, all_times = self._session.read_event(
            self._session.active_event
        )

        points = np.asarray(points, dtype=np.float32)
        colors = np.asarray(colors, dtype=np.float32)
        if hasattr(self, "_source_plot"):
            self._source_plot.setData(pos=points, color=colors)


###SPECTROGRAM VIEWER##
    def _setup_spec_viewer(self):
        spec_viewer = getattr(self, "specViewer", None)
        if spec_viewer is None:
            logger.warning("Widget 'specViewer' non trovato nella UI.")
            return

        spec_image_height = 129
        spec_image_width = 500
        self.spec_image_data = np.zeros(
            (spec_image_height, spec_image_width), dtype=np.uint8
        )

        self.spectrogram = pg.ImageItem()
        spec_viewer.addItem(self.spectrogram)
        self.spectrogram.setImage(self.spec_image_data)
        self.spectrogram.setOpts(axisOrder="row-major")

        axis = spec_viewer.getAxis("left")
        axis.setTicks([[(0, "10k"), (64, "50k"), (128, "90k")]])

    def _update_spec_viewer(self):
        if self._session.active_event is not None:
            ev = self._session.active_event
            available_data = min(
                len(ev.spectrogram), np.shape(self.spec_image_data)[1]
            )

            if available_data > 0:
                self.spec_image_data.fill(0)
                data = np.array(
                    ev.spectrogram[-available_data:], dtype=np.uint8
                )
                self.spec_image_data[:, :available_data] = data.T

                self.spectrogram.setImage(self.spec_image_data)

####VU METER VIEWER##
    def _setup_vu_meter(self):
        vu_meter = getattr(self, "VUMeter", None)
        
        if vu_meter is None:
            logger.warning("Widget 'VUMeter' not found in UI.")
            return
        
        # Hide axes and disable mouse interaction (panning/zooming)
        vu_meter.hideAxis("bottom")
        vu_meter.hideAxis("left")
        vu_meter.setMouseEnabled(x=False, y=False)
        
        # Strict range setup: Y from 0 (min dB) to 1 (full scale / 0 dB)
        vu_meter.setYRange(0, 1, padding=0)
        vu_meter.setXRange(0, 1, padding=0)
        vu_meter.disableAutoRange()
        
        # Define dB limits for visualization
        self.min_db = -60.0  # Floor (Y = 0.0)
        self.max_db = 0.0    # Ceiling / Full Scale (Y = 1.0)
        
        # Create the VU bar item pinned at y0=0
        self.vu_bar = pg.BarGraphItem(
            x=[0.5], 
            y0=[0], 
            height=[0.3], 
            width=0.8, 
            brush="g"
        )
        vu_meter.addItem(self.vu_bar)
        
        # Convert linear threshold directly to normalized [0, 1] GUI position
        norm_thresh = self._rms_to_norm(self._state.threshold)
        
        # Create the threshold line
        self.threshold_line = pg.InfiniteLine(
            angle=0, 
            pos=norm_thresh, 
            pen=pg.mkPen("r", width=2)
        )
        vu_meter.addItem(self.threshold_line)



    def _update_vu_meter(self):
        if hasattr(self, "vu_bar"):
            level = self._state.EMA_rms
            self.vu_bar.setOpts(height=[self._rms_to_norm(level)])
            self.threshold_line.setPos(self._rms_to_norm(self._state.threshold))

##BUTTON CALLBACKS##
  
    def _on_play_pause_clicked(self):
        logger.info("Pulsante Play/Pause premuto")

    def _on_slowmo_clicked(self):
        logger.info("Pulsante 0.5x premuto")

    def _on_stop_clicked(self):
        logger.info("Pulsante STOP premuto")

    def _on_mode_toggle_clicked(self):
        self._state.is_live = not self._state.is_live
        if self._state.is_live:
            self.ToggleButton.setText("Live")
            self.ToggleButton.setStyleSheet("background-color: green; color: white;")
        else:
            self.ToggleButton.setText("Playback")
            self.ToggleButton.setStyleSheet("background-color: orange; color: black;")


        logger.info("Pulsante Live/Playback premuto")

    def _on_threshold_changed(self, value):
        # Convert slider value to RMS threshold
        self._state.threshold = value/140.0 + 0.005
        logger.debug(f"Soglia impostata a: {self._state.threshold:.3f} (slider value: {value})")

    def _rms_to_norm(self, rms_val, eps=1e-6):
        """Converts a linear RMS value directly to a normalized [0, 1] scale in dB space."""
        # 1. Convert linear RMS to dBFS
        db_val = 20.0 * np.log10(max(rms_val, eps))
        
        # 2. Map dB value from [min_db, max_db] to [0.0, 1.0]
        norm = (db_val - self.min_db) / (self.max_db - self.min_db)
        
        # 3. Clamp output between 0.0 and 1.0
        return float(np.clip(norm, 0.0, 1.0))    

    def stop(self):
        logger.info("Arresto della GUI e chiusura sessione...")
        self._session.kill_event()
        self._state.stop(__name__)
        self._poller.stop()
        time.sleep(1)
        QApplication.quit()