from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QFormLayout, QLabel, QGroupBox, QFileDialog, QSizePolicy, QFrame
from PyQt5.QtGui import QKeySequence, QPixmap
from PyQt5.QtCore import Qt, QSettings, QDir, pyqtSignal

from PIL import Image, ImageCms
from PIL.ImageQt import ImageQt


class ImageViewer(QLabel):

    selectionCapturedSignal = pyqtSignal()

    def __init__(self, fn_icc_input=None, fn_icc_output=None, input_colorsystem="RGB", output_colorsystem="RGB"):
        super().__init__(None)
        #self.setMouseTracking(True)
        self.imagePixmap = None
        self.imageSelection = None
        self.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.setStyleSheet("background-color: white;")
        self.resetCapture()

        self.transform = None
        if fn_icc_input and fn_icc_output:
            input_profile = ImageCms.getOpenProfile(fn_icc_input)
            output_profile = ImageCms.getOpenProfile(fn_icc_output)
            self.transform = ImageCms.ImageCmsTransform(input_profile, output_profile, input_mode=input_colorsystem, output_mode=output_colorsystem)

    def resetCapture(self):
        self.start_x = -1
        self.start_y = -1
        self.end_x = -1
        self.end_y = -1

    def mousePressEvent(self, event):
        if self.imagePixmap:
            if event.x() <= self.imagePixmap.width() and event.y() <= self.imagePixmap.height():
                self.start_x = event.x()
                self.start_y = event.y()

    def mouseReleaseEvent(self, event):
        self.end_x = event.x()
        self.end_y = event.y()
        if self.imagePixmap:
            if self.end_x <= self.imagePixmap.width() and self.end_y <= self.imagePixmap.height() and self.start_x >= 0 and self.start_y >= 0:
                self.captureSelection()
            else:
                self.resetCapture()

    def captureSelection(self):
        if self.start_x > self.end_x:
            width = self.start_x - self.end_x
            x = self.end_x
        else:
            width = self.end_x - self.start_x
            x = self.start_x

        if self.start_y > self.end_y:
            height = self.start_y - self.end_y
            y = self.end_y
        else:
            height = self.end_y - self.start_y
            y = self.start_y

        if self.start_x == self.end_x or self.start_y == self.end_y:
            return

        self.imageSelection = self.imagePixmap.copy(x, y, width, height)
        self.selectionCapturedSignal.emit()

    def getSelection(self):
        return self.imageSelection

    def resizeEvent(self, event):
        if self.imagePixmap:
            self.imagePixmap.scaled(self.width(), self.height(), Qt.KeepAspectRatio)

    def loadImage(self, fileName):
        img = Image.open(fileName)
        if self.transform:
            img = ImageCms.applyTransform(img, self.transform)
        qimg = ImageQt(img)
        self.imagePixmap = QPixmap.fromImage(qimg).scaled(self.width(), self.height(), Qt.KeepAspectRatio)
        self.setPixmap(self.imagePixmap)
