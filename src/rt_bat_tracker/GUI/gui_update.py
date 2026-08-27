from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import QTimer
from PyQt5.uic import loadUi
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np
import sys
import time
import logging

logger = logging.getLogger("GUIUPDATE")
logger.setLevel(logging.INFO)


def run(state, cfg, session):
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import Qt

    # must be set before QApplication is created
    QApplication.setAttribute(Qt.AA_UseDesktopOpenGL)
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)

    app = QApplication(sys.argv)

    window = MainWindow(state, cfg, session)
    window.show()

    app.aboutToQuit.connect(window.stop)  # window closed → stops all threads

    app.exec_()  # blocks until window closes
    state.gui_running_flag = False


class MainWindow(QMainWindow):
    def __init__(self, state, cfg, session):
        super().__init__()
        import pyqtgraph as pg

        try:
            loadUi(
                cfg.GUIpath,
                self,
                {
                    "GLViewWidget": gl.GLViewWidget,
                    "PlotWidget": pg.PlotWidget,
                    "GraphicsLayoutWidget": pg.GraphicsLayoutWidget,
                },
            )

        except Exception as e:
            logger.error("unable to open the GUI Layout: %s \n%s ", cfg.GUIpath, e)
            state.stop(__name__)
            return

        self._session = session
        self._state = state
        self._cfg = cfg
        # self.event = None
        self.timer = int(1000 / self._cfg.update_fps)
        self.setWindowTitle(self._session.session_name)

        try:
            self._setup_path_viewer(self._state.micxyz)
            self._setup_spec_viewer()
        except Exception as e:
            logger.error("unable to initialize pathViewer: ", e)
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

        self._session.write_audiofile()
        self._update_path_viewer()
        self._update_spec_viewer()

    def _update_path_viewer(self):
        """
        takes results from the queue
        updates the session/event adding new points
        plots all the points existing in the event by now
        """
        pos, timestamp = self._state.get_result()
        logger.debug(f"GUI received this point: {pos} [timestamp: {timestamp}]")
        self._session.update(pos, timestamp)
        points, colors, all_times = self._session.read_event(self._session.active_event)
        # there was some compatibility issue with setData, maybe try to store points differently in the session
        points = np.asarray(points, dtype=np.float32)
        colors = np.asarray(colors, dtype=np.float32)
        self._source_plot.setData(pos=points, color=colors)

    def _setup_path_viewer(self, micxyz):
        """
        initializes the main graph that plots the real time position of the bat tracked
        is then updated in update function
        """

        grid = gl.GLGridItem()
        grid.setSize(20, 20)
        grid.setSpacing(0.5, 0.5)
        self.pathViewer.addItem(grid)

        self._mic_plot = gl.GLScatterPlotItem(
            pos=micxyz,
            color=(1, 0, 0, 1),
            size=10,
        )
        self.pathViewer.addItem(self._mic_plot)

        self._source_plot = gl.GLScatterPlotItem(
            pos=np.array([[0, 0, 0]], dtype=np.float64),
            color=(1, 1, 0, 1),
            size=10,
        )
        self.pathViewer.addItem(self._source_plot)

        self.pathViewer.setCameraPosition(distance=20, azimuth=-50, elevation=30)

    def _update_spec_viewer(self):
        """Updates the spectrogram viewer with the latest data from the active event."""
        if self._session.active_event is not None:
            ev = self._session.active_event
            available_data = min(len(ev.spectrogram), np.shape(self.spec_image_data)[1])

            if available_data > 0:
                self.spec_image_data.fill(0)
                data = np.array(ev.spectrogram[-available_data:], dtype=np.uint8)
                self.spec_image_data[:, :available_data] = data.T

                self.spectrogram.setImage(
                    self.spec_image_data,
                )

    def _setup_spec_viewer(self):
        """Initializes the spectrogram viewer with a blank image and sets up the axes."""
        spec_image_height = 129
        spec_image_width = 500  # 1081
        self.spec_image_data = np.zeros(
            (spec_image_height, spec_image_width), dtype=np.uint8
        )

        self.spectrogram = pg.ImageItem()

        self.specViewer.addItem(self.spectrogram)

        self.spectrogram.setImage(self.spec_image_data)  # correct levels in case

        self.spectrogram.setOpts(axisOrder="row-major")  # or col-major

        axis = self.specViewer.getAxis("left")

        axis.setTicks(
            [
                [
                    (0, "10k"),
                    (64, "50k"),
                    (128, "90k"),
                ]
            ]
        )

    def stop(self):
        print("STOOOOP!!!!")
        self._session.kill_event()
        self._state.stop(__name__)
        self._poller.stop()
        time.sleep(2)
        QApplication.quit()
