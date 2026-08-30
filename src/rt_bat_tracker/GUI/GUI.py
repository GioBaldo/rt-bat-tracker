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
    window.showFullScreen()

    app.aboutToQuit.connect(window.stop)

    app.exec_()
    state.gui_running_flag = False

##MAIN WINDOW CLASS##
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
        self.selected_event_index = 0
        self.playback_speed = 1
        self.playback_counter = 0
        self.pause_counter = False
        # import widgets and connect signals
        self._import_ui_widgets()
        self._connect_ui_signals()
        self._live_layout()  # set initial layout to live mode
        self.TextLabel.setText(f"{self._session.session_name}")
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
        if self._state.is_live:
            self._session.write_audiofile()
            self._update_path_viewer()
            self._update_spec_viewer()
            self._update_vu_meter()
            self._update_list_viewer()
        else:
            try:
                self.selected_event = self._session.event_list[self.selected_event_index]
            except IndexError:
                logger.info("Selected event index out of bounds for events list. [or no events available]")
                time.sleep(1)
                self._state.is_live = True
                self._live_layout()
                return
            self.playback_time = 1/self._cfg.update_fps * self.playback_speed * self.playback_counter
            if self.playback_time >= self.selected_event.duration: 
                self.playback_counter = 0  
                self.playback_time = 0          
            self._show_selected_event(self.selected_event, self.playback_time)
            if not self.pause_counter:
                self.playback_counter += 1
            
###VIEWERS FUNCTIONS##
###PATH VIEWER###
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

        yaxis = spec_viewer.getAxis("left")
        yaxis.setTicks([[(0, "10k"), (64, "50k"), (128, "90k")]])

        xaxis = spec_viewer.getAxis("bottom")
        unity = self._cfg.HOP_SIZE / self._cfg.fs
        xaxis.setScale(unity)
        #xaxis.setTicks([[(0, "0s"), (250, "2.5s"), (500, "5s")]])

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

###VU METER VIEWER##
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
###LIST VIEWER##
    def _update_list_viewer(self):
        if self._session.active_event is not None: 
            item_count = self.ListViewer.count()
            last_item = self.ListViewer.item(item_count - 1) if item_count > 0 else None
            if last_item is None or last_item.text() != self._session.active_event.event_name:
                self.ListViewer.addItem(self._session.active_event.event_name)
                self.ListViewer.scrollToBottom()

###PLAYBACK FUNCTIONS##
    def _show_selected_event(self, event, playback_time):
        """
        Given the event and the time in seconds this fucntion shows all the points in the event before the playback_time
        and the spectrogram accordingly.
        """
        points, colors = self._session.read_points_for_playback(event, playback_time)
        points = np.asarray(points, dtype=np.float32)
        colors = np.asarray(colors, dtype=np.float32)
        self._source_plot.setData(pos=points, color=colors)

        self.spec_image_data = self._session.read_spectrogram_for_playback(event, playback_time, self.spec_image_data.shape)
        self.spectrogram.setImage(self.spec_image_data)

##BUTTON CALLBACKS##
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
        self._connect_signal_safely(
            "listWidget", "currentRowChanged", self._on_event_selected
        )

    def _import_ui_widgets(self):
        """Importa i widget dalla UI e li assegna come attributi della classe."""
        self.ExitButton = getattr(self, "ExitButton", None)
        self.PlayButton = getattr(self, "PlayButton", None)
        self.LeftButton = getattr(self, "LeftButton", None)
        self.RightButton = getattr(self, "RightButton", None)
        self.ToggleButton = getattr(self, "pushButton_4", None)
        self.ThresholdSlider = getattr(self, "ThresholdSlider", None)
        self.ListViewer = getattr(self, "listWidget", None)
        self.TextLabel = getattr(self, "label", None)

        self.ExitButton.setStyleSheet("background-color: red; color: white;")
        self.ThresholdSlider.installEventFilter(self)  # Install event filter for mouse wheel events

    def eventFilter(self, source, event):
        """Handle mouse hover events (for ThresholdSlider)"""
        if source == self.ThresholdSlider:
            if event.type() == event.Enter or event.type() == event.MouseMove or event.type() == event.MouseButtonPress:
                self.TextLabel.setText(f"Threshold: {self._state.threshold:.3f}")
            elif event.type() == event.Leave:
                self.TextLabel.setText(f"{self._session.session_name}")
        return super().eventFilter(source, event)

    def _on_event_selected(self, item_idx):
        self.selected_event_index = item_idx

    def _on_play_pause_clicked(self):
        if self._state.is_live:
            self._state.SAVE_RESULTS = not self._state.SAVE_RESULTS
            if self._state.SAVE_RESULTS:
                self.PlayButton.setText("Saving ON")
                self.PlayButton.setStyleSheet("background-color: green; color: white;")
            else:
                self.PlayButton.setText("Saving OFF")
                self.PlayButton.setStyleSheet("background-color: red; color: black;")
        else:
            self.pause_counter = not self.pause_counter
            if self.pause_counter:
                self.PlayButton.setText("Play")
                self.PlayButton.setStyleSheet("background-color: yellow; color: black;")
            else:
                self.PlayButton.setText("Pause")
                self.PlayButton.setStyleSheet("background-color: green; color: white;")

    def _on_slowmo_clicked(self):
        logger.info("Pulsante 0.5x premuto")

    def _on_stop_clicked(self):
        logger.info("Pulsante STOP premuto")

    def _on_mode_toggle_clicked(self):
        self._state.is_live = not self._state.is_live
        if self._state.is_live:
            self._live_layout()
        else:
            self._playback_layout()
            self._session.kill_event()  # Terminate event before changing mode

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

    def _live_layout(self):
        self.ToggleButton.setText("Live")
        self.ToggleButton.setStyleSheet("background-color: green; color: white;")
        self.RightButton.setEnabled(False)
        self.LeftButton.setEnabled(False)
        self.PlayButton.setEnabled(True)
        self.PlayButton.setText("Saving ON" if self._state.SAVE_RESULTS else "Saving OFF")
        self.PlayButton.setStyleSheet("background-color: green; color: white;" if self._state.SAVE_RESULTS else "background-color: red; color: black;")
        self.ThresholdSlider.setEnabled(True)
        self.ListViewer.setEnabled(False)
    
    def _playback_layout(self):
        self.ToggleButton.setText("Playback")
        self.ToggleButton.setStyleSheet("background-color: orange; color: black;")
        self.RightButton.setEnabled(True)
        self.LeftButton.setEnabled(True)
        self.PlayButton.setEnabled(True)
        self.PlayButton.setText("Pause")
        self.pause_counter = False
        self.PlayButton.setStyleSheet("background-color: green; color: white;")
        self.ThresholdSlider.setEnabled(False)
        self.ListViewer.setEnabled(True)

    def stop(self):
        logger.info("Arresto della GUI e chiusura sessione...")
        self._session.kill_event()
        self._state.stop(__name__)
        self._poller.stop()
        time.sleep(1)
        QApplication.quit()