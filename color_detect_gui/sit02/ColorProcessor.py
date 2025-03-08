import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')

from PySide6.QtCore import Slot, Signal

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

import LimhesColor as lc


class ColorProcessor(FigureCanvasQTAgg):

    color_added = Signal(int, np.ndarray, np.ndarray, np.ndarray)

    def __init__(self, parent=None, width=5, height=10, dpi=100):
        self.color_array = None
        self.data_names = ["ab", "La", "Lb"]

        self.data_ref = {} # "identifier": [axis_references ...]

        self.fig = Figure(figsize=(width, height), dpi=dpi, constrained_layout = True)
        #self.fig = Figure(constrained_layout = True)
        self.axes = []
        self.charting_indices = [{"x": 0, "y": 1}, {"x": 0, "y": 2}, {"x": 1, "y": 2}]
        self.axis_labels = ["L", "a", "b"]
        for axis in range(0, 3):
            self.axes.append( self.fig.add_subplot(1, 3, 1+axis) )
            self.axes[axis].set_xlabel(self.axis_labels[self.charting_indices[axis]["x"]])
            self.axes[axis].set_ylabel(self.axis_labels[self.charting_indices[axis]["y"]])

        super().__init__(self.fig)

    @Slot()
    def addColor(self, identifier, color_array):
        self.color_array = color_array
        color_mean = np.mean(self.color_array, axis=0)
        rgb_mean = color_mean[0:3]

        rgb = self.color_array[:,0:3]
        lab = lc.rgb2lab(rgb, "D50")
        lab_mean = np.mean(lab, axis=0)
        lab_stdev = np.std(lab, axis=0)

        self.data_ref[str(identifier)] = []
        for axis in range(0, 3):
            self.data_ref[str(identifier)].append(
                self.axes[axis].scatter(lab[:,self.charting_indices[axis]["x"]],
                                        lab[:,self.charting_indices[axis]["y"]],
                                        s=0.1, c=[rgb_mean/256.0])
            )
            self.data_ref[str(identifier)].append(
                self.axes[axis].annotate(str(identifier), (lab_mean[self.charting_indices[axis]["x"]],
                                                           lab_mean[self.charting_indices[axis]["y"]]),
                                                           fontsize=15, color='black')
            )
            self.fig.canvas.draw()

        self.color_added.emit(identifier, lab_mean, lab_stdev, rgb_mean)

    @Slot()
    def removeColor(self, identifier):
        for ref in self.data_ref[str(identifier)]:
            ref.remove()
        self.fig.canvas.draw()
