import numpy as np

from PySide6.QtWidgets import QGraphicsScene, QGraphicsView, QHBoxLayout, QLabel, QPushButton, QFileDialog, QWidget
from PySide6.QtGui import QPixmap, QImage, QPen, QFont
from PySide6.QtCore import Qt, QRectF, QRect, Signal, Slot

from MouseHandler import MouseHandler, MouseState


class ImageLoader(QWidget):

    file_selected = Signal(str)

    def __init__(self, parent=None, settings=None):
        QWidget.__init__(self, parent)

        self.settings = settings

        lbl = QLabel("Use the mouse wheel to zoom. Right-mouse-click-drag to move the image around. Draw a rectangle with the left mouse button to select a color region.")
        lbl.setWordWrap(True)

        select_file_button = QPushButton("Load image from file")
        select_file_button.clicked.connect(self.selectFile)

        layout = QHBoxLayout()
        self.setLayout(layout)
        self.layout().addWidget(lbl)
        self.layout().addWidget(select_file_button)

    @Slot()
    def selectFile(self):
        file_open_path = self.settings.value("General/fileOpenPath", ".")
        file_name = QFileDialog.getOpenFileName(self, "Open Image", file_open_path , "Image Files (*.png *.jpg *.jpeg *.bmp *tif *tiff)")
        if not file_name[0]:
            return
        self.settings.setValue("General/fileOpenPath", file_name[0])
        print(file_name[0])
        self.file_selected.emit(file_name[0])


class ScrollableImage(MouseHandler, QGraphicsView):

    color_added = Signal(int, np.ndarray)

    def __init__(self, parent=None):
        QGraphicsView.__init__(self, parent)
        self.setMouseTracking(True)

        MouseHandler.__init__(self, parent_class=QGraphicsView)
        self.add_mouse_handler(
            event={'event': 'move', 'button': 'right'},
            callback=self.scrollView,
            args=[MouseState('delta_x'), MouseState('delta_y')]
        )
        self.add_mouse_handler(
            event={'event': 'move', 'button': 'left'},
            callback=self.drawTemporaryRectangle,
            args=[MouseState('pressed_x'), MouseState('pressed_y'), MouseState('x'), MouseState('y')]
        )
        self.add_mouse_handler(
            event={'event': 'release', 'button': 'left'},
            callback=self.addRectangle,
            args=[]
        )
        self.add_mouse_handler(
            event={'event': 'wheel'},
            callback=self.zoomImage,
            args=[MouseState('delta_x'), MouseState('delta_y'), MouseState('wheel_delta')]
        )

        # initialization values:
        self.item_pixmap = None
        self.loaded_qimage = None
        self.selection_qrect = None
        self.color_array = []

        self.item_types =  ["rectangles", "texts", "indices"]
        self.graphicsitems = {"rectangles": [], "texts": [], "indices": []}

        self.pen = QPen(Qt.green, 3, Qt.DashDotLine, Qt.RoundCap, Qt.RoundJoin)
        self.font = QFont()
        self.font.setPointSize(150)

        if self.scene() == None:
            self.setScene(QGraphicsScene())

        #self.loadImage("/home/reneb/git/ColorIdentification/color_detect_gui_2/scrollable_image_test/sit02/resources/epson1200_builtinCCT_estoniablumenmuster.bmp")
        #self.loadImage("/home/reneb/git/ColorIdentification/color_detect_gui_2/scrollable_image_test/sit02/resources/ri1.jpeg")

        self.item_selection_rectangle = self.scene().addRect(QRect(0,0,0,0), self.pen, Qt.NoBrush)
        self.item_selection_rectangle.setZValue(1)

    @Slot()
    def loadImage(self, file_name):
        self.loaded_qimage = QImage(file_name)
        if self.item_pixmap == None:
            self.item_pixmap = self.scene().addPixmap(QPixmap.fromImage(self.loaded_qimage))
        else:
            self.item_pixmap.setPixmap(QPixmap.fromImage(self.loaded_qimage))
        self.item_pixmap.setZValue(0)
        self.fitInView(self.item_pixmap, aspectRadioMode=Qt.KeepAspectRatio)

    @Slot()
    def scrollView(self, delta_x, delta_y):
        h = self.horizontalScrollBar()
        v = self.verticalScrollBar()
        h.setValue(h.value() + delta_x)
        v.setValue(v.value() + delta_y)

    @Slot()
    def drawTemporaryRectangle(self, pressed_x, pressed_y, x, y):
        max_x, max_y = max(pressed_x, x), max(pressed_y, y)
        min_x, min_y = min(pressed_x, x), min(pressed_y, y)
        self.selection_qrect = self.mapToScene(QRectF(min_x, min_y, (max_x-min_x), (max_y-min_y)).toRect()).boundingRect()
        self.selection_qrect = self.selection_qrect & self.loaded_qimage.rect() # trim selection by intersecting (using operator &)

        self.item_selection_rectangle.show()
        self.item_selection_rectangle.setRect(self.selection_qrect)
        self.item_selection_rectangle.update()

    @Slot()
    def addRectangle(self):
        self.item_selection_rectangle.hide()

        new_index = self.graphicsitems["indices"][-1] + 1 if len(self.graphicsitems["indices"]) > 0 else 1
        self.graphicsitems["indices"].append( new_index )

        rect = self.scene().addRect(self.selection_qrect, self.pen, Qt.NoBrush)
        rect.setZValue(1)
        self.graphicsitems["rectangles"].append( rect )

        stamp_count_text = self.scene().addText(str(new_index), self.font)
        stamp_count_text.setPos(self.selection_qrect.center())
        stamp_count_text.setZValue(1)
        self.graphicsitems["texts"].append( stamp_count_text )

        qimage_cropped = self.loaded_qimage.copy(self.selection_qrect.toRect()).convertToFormat(QImage.Format_RGBX8888)
        self.color_array = np.array(qimage_cropped.constBits(), copy=True).reshape(qimage_cropped.height()*qimage_cropped.width(), 4)
        self.color_added.emit(new_index, self.color_array)

    @Slot()
    def removeRectangle(self, index):
        list_index = self.graphicsitems["indices"].index(index)
        for type in self.item_types:
            if not type == "indices":
                self.scene().removeItem( self.graphicsitems[type][list_index] )
            self.graphicsitems[type].pop(list_index)

    @Slot()
    def zoomImage(self, delta_x, delta_y, wheel_delta):
        zoom_factor = 1.25
        zoom_factor = zoom_factor if wheel_delta.y() > 0 else 1/zoom_factor
        self.scale(zoom_factor, zoom_factor)
        self.translate(delta_x, delta_y)
