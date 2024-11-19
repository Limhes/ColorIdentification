from PySide6.QtWidgets import QTableWidgetItem, QLabel, QWidget, QVBoxLayout
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt

from stampcolor import ColorTransform


class TableItem(QTableWidgetItem):
    def __init__(self, parent=None):
        super().__init__(parent)

    def setLab(self, Lab):
        self.Lab = Lab

    def getLab(self):
        return self.Lab


class PictureViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.image_pixmap = None
        self.colorTransform = ColorTransform()

        self.lblPicture = QLabel()
        self.lblInformation = QLabel()
        self.lblInformation.setFixedHeight(50)
        layout = QVBoxLayout()
        layout.addWidget(self.lblPicture)
        layout.addWidget(self.lblInformation)
        self.setLayout(layout)

    def setImageQt(self, image_qt):
        self.image_qt = image_qt
        self.image_pixmap = QPixmap.fromImage(image_qt).scaled(self.lblPicture.width(), self.lblPicture.height(), Qt.KeepAspectRatio)
        self.image_qt_scaled = self.image_pixmap.toImage()
        self.lblPicture.setPixmap(self.image_pixmap)

    def mousePressEvent(self, event):
        if self.image_pixmap:
            self.color_picker_pos = self.lblPicture.mapFromParent( event.pos() )
            if self.color_picker_pos.x() >= 0 and self.color_picker_pos.y() >= 0 \
                and self.color_picker_pos.x() <= self.image_pixmap.size().width() and self.color_picker_pos.y() <= self.image_pixmap.size().height():

                pixel_qcolor = self.image_qt_scaled.pixelColor(self.color_picker_pos)
                color = self.colorTransform.colorDict([ pixel_qcolor.red(), pixel_qcolor.green(), pixel_qcolor.blue() ])

                self.lblInformation.setText( f"RGB: {color['rgb']}, CIEXYZ (D65, 2°): {color['ciexyz']}, CIELAB (D65, 2°): {color['cielab']}" +
                    f"\nMatch: Stanley-Gibbons: {color['colorkey']}\nMatch: Munsell: {color['munsell']}" )

