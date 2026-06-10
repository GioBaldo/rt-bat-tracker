from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import QTimer
from PyQt5.uic import loadUi
import pyqtgraph.opengl as gl
import numpy as np
import sys
import time
import logging

logger = logging.getLogger(__name__)


def run(state, cfg):
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import Qt

    # must be set before QApplication is created
    QApplication.setAttribute(Qt.AA_UseDesktopOpenGL)
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)

    app = QApplication(sys.argv)

    window = MainWindow(state, cfg)
    window.show()

    app.aboutToQuit.connect(state.stop)  # window closed → stops all threads

    app.exec_()  # blocks until window closes
    state.gui_running_flag = False


class MainWindow(QMainWindow):
    def __init__(self, state, cfg):
        super().__init__()
        import pyqtgraph as pg

        # pg.setConfigOption("qt_lib", "PyQt5")

        try:
            loadUi(cfg.GUIpath, self, {"GLViewWidget": gl.GLViewWidget})
        except Exception as e:
            logger.error("unable to open the GUI Layout: %s \n ", cfg.GUIpath, e)
            state.stop(__name__)
            return

        self._state = state
        self._cfg = cfg
        self.timer = int(1000 / self._cfg.update_fps)
        try:
            self._setup_path_viewer(self._state.micxyz)
        except Exception as e:
            logger.error("unable to initialize pathViewer: ", e)
            state.stop(__name__)

        self._poller = QTimer(self)
        self._poller.setInterval(self.timer)  # 20fps
        self._poller.timeout.connect(self._update)
        self._poller.start()

    def _update(self):
        if self._state.stop_event.isSet():
            return
        if not self._state.gui_running_flag:
            logger.info(
                "GUI LOADED: time elapsed for loading - %4f",
                time.time() - self._state.gui_t_start,
            )
            self._state.gui_running_flag = True

        result, timestamp = self._state.get_result()
        # if result == None or result == []:
        # print(f"strange result type {type(result)}")
        # return

        self._state.write_buffer(result, timestamp)
        points, all_times = self._state.read_buffer()

        logger.info("GUI updated succesfully with new results: %s", type(points))
        # update widgets here

        self._source_plot.setData(pos=points)

    def _setup_path_viewer(self, micxyz):
        # no initializeGL() call

        grid = gl.GLGridItem()
        grid.setSize(10, 10)
        grid.setSpacing(1, 1)
        self.pathViewer.addItem(grid)  # ← was missing

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
