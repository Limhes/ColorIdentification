import sys

from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget
from PySide6.QtCore import QSettings

# Important:
# You need to run the following command to generate the ui_form.py file
#     pyside6-uic form.ui -o ui_form.py

from ui_form import Ui_MainWindow
from ScrollableImage import ScrollableImage, ImageLoader
from ColorProcessor import ColorProcessor
from ColorList import ColorList

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.settings = QSettings("config.ini", QSettings.IniFormat)

        # setup left panel:
        imageLoader = ImageLoader(self, self.settings)
        scrollableImage = ScrollableImage()
        colorProcessor = ColorProcessor()

        self.wdgLeft = QWidget()
        self.wdgLeft.setLayout( QVBoxLayout() )
        self.wdgLeft.layout().addWidget( imageLoader )
        self.wdgLeft.layout().addWidget( scrollableImage )
        self.wdgLeft.layout().addWidget( colorProcessor )
        self.wdgLeft.layout().setStretch(0, 1)
        self.wdgLeft.layout().setStretch(1, 7)
        self.wdgLeft.layout().setStretch(2, 4)

        # setup right panel:
        add_stamp_button = QPushButton("Remove selected color")
        colorList = ColorList()

        #self.ui.tabRight.setFixedWidth(250)
        self.wdgRight = QWidget()
        self.wdgRight.setLayout( QVBoxLayout() )
        self.wdgRight.layout().addWidget( add_stamp_button )
        self.wdgRight.layout().addWidget( colorList )

        # setup main window:
        self.ui.centralwidget.layout().addWidget( self.wdgLeft )
        self.ui.centralwidget.layout().addWidget( self.wdgRight )
        self.ui.centralwidget.layout().setStretch(0, 3)
        self.ui.centralwidget.layout().setStretch(1, 1)

        # connect signals & slots:
        imageLoader.file_selected.connect( scrollableImage.loadImage )
        scrollableImage.color_added.connect( colorProcessor.addColor )
        colorProcessor.color_added.connect( colorList.addColor )
        add_stamp_button.clicked.connect( colorList.removeSelectedRow )
        colorList.remove_color.connect( colorProcessor.removeColor )
        colorList.remove_color.connect( scrollableImage.removeRectangle )

if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = MainWindow()
    #widget.show()
    widget.showMaximized()
    sys.exit(app.exec())
