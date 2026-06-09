from PyQt5.QtWidgets import *
from PyQt5 import uic


class Window:

    def __ini__(self):
        super(Window, self).__init__()
        uic.loadUi("GUI_layout", self)
        self.show()
