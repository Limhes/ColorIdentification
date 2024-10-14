#!/usr/bin/env python3

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QFormLayout, QLabel, QGroupBox, QFileDialog, QSizePolicy, QFrame
from PyQt5.QtGui import QKeySequence, QPixmap, QImage
from PyQt5.QtCore import Qt, QSettings, QDir, pyqtSignal, QByteArray, QBuffer, QIODevice


from stampcolor.widgets import ImageViewer
from stampcolor.color_func import colorTransform


class MainWindow(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # set the window title
        self.setWindowTitle('Hello World')

        self.setupSettings()
        self.setupUI()

        self.colorTransform = colorTransform()

        # show the window
        self.showMaximized()

    def setupSettings(self):
        self.settings = QSettings("config.ini", QSettings.IniFormat)

    def setupUI(self):
        # control layout
        controlFrame = QFrame()
        controlFrame.setFrameStyle(QFrame.Panel | QFrame.Raised)
        controlFrame.setLineWidth(3)
        controlLayout = QFormLayout()
        controlFrame.setLayout(controlLayout)
        # control items
        buttonLoadImage = QPushButton("Load image from file")
        buttonLoadImage.clicked.connect(self.loadImageFromFile)
        controlLayout.addRow("Source image:", buttonLoadImage)
        self.keyPressText = QLabel()
        controlLayout.addRow("Key pressed:", self.keyPressText)
        self.selectionLabel = QLabel()
        controlLayout.addRow("Selection captured: ", self.selectionLabel)
        colorFrame = QFrame()
        self.colorsLayout = QVBoxLayout()
        colorFrame.setLayout(self.colorsLayout)
        controlLayout.addRow("Colors detected: ", colorFrame)

        # image label
        self.imageViewer = ImageViewer(self.settings.value("ColorTransform/fileNameInputProfile", None),
                                       self.settings.value("ColorTransform/fileNameOutputProfile", None),
                                       self.settings.value("ColorTransform/inputColorSystem", "RGB"),
                                       self.settings.value("ColorTransform/outputColorSystem", "RGB"))
        self.imageViewer.selectionCapturedSignal.connect(self.selectionCaptured)
        # image layout
        imageFrame = QFrame()
        imageLayout = QVBoxLayout()
        imageLayout.addWidget(self.imageViewer)
        imageFrame.setLayout(imageLayout)

        # main layout:
        mainLayout = QHBoxLayout()
        self.setLayout(mainLayout)
        mainLayout.addWidget(imageFrame)
        mainLayout.addWidget(controlFrame)

    def keyPressEvent(self, e):
        key = QKeySequence(e.key()).toString().lower()
        self.keyPressText.setText(key)
        try:
            keyNumber = int(key)
            if keyNumber > 1 and keyNumber < 10:
                for i in reversed(range(self.colorsLayout.count())):
                    self.colorsLayout.itemAt(i).widget().setParent(None) # remove from layout

                for color in self.colorTransform.cluster(self.selectionLabel.pixmap(), keyNumber):
                    lbl = QLabel()
                    lbl.setFixedHeight(150)
                    lbl.setFixedWidth(500)
                    #lbl.setText(f"RGB: {color['rgb']}\nMunsell: {color['munsell']}\nISCC-NBS: {color['isccnbs']}\nStanley-Gibbons: {color['colorkey']}")
                    lbl.setText(f"RGB: {color['rgb']}\nCIEXYZ (D65, 2°): {color['ciexyz']}\nCIELAB (D65, 2°): {color['cielab']}\n" +
                                f"\nMatch: Stanley-Gibbons: {color['colorkey']}\nMatch: Munsell: {color['munsell']}")
                    lbl.setStyleSheet(f"background-color: rgb({color['rgb']})")
                    self.colorsLayout.addWidget(lbl)

        except ValueError as e:
            pass

    def loadImageFromFile(self):
        fileOpenPath = self.settings.value("General/fileOpenPath", ".")
        fileName = QFileDialog.getOpenFileName(self, "Open Image", fileOpenPath , "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if fileName[0]:
            self.settings.setValue("General/fileOpenPath", fileName[0])
            self.imageViewer.loadImage(fileName[0])

    def selectionCaptured(self):
        self.selectionLabel.setPixmap(self.imageViewer.getSelection())
        #self.selectionLabel.setText("rectangle " + str(self.imageViewer.getSelection().width()) + "x" + str(self.imageViewer.getSelection().height()) + " px")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec())
