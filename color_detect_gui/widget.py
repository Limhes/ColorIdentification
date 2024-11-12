# This Python file uses the following encoding: utf-8
import sys

from PySide6.QtWidgets import QApplication, QWidget, QFileDialog, QTableWidgetItem, QHeaderView
from PySide6.QtGui import QPixmap, QImage, QColor
from PySide6.QtCore import Qt, QSettings

from PIL import Image, ImageCms
import cv2 as cv
import numpy as np
import sane

from stampcolor.color_func import colorTransform, unsharp_mask, crop_rect

# Important:
# You need to run the following command to generate the ui_form.py file
#     pyside6-uic form.ui -o ui_form.py, or
#     pyside2-uic form.ui -o ui_form.py
from ui_form import Ui_Widget


class Widget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_Widget()
        self.ui.setupUi(self)

        self.ui.btnLoadFile.clicked.connect(self.loadImageFromFile)
        self.ui.btnFindStamps.clicked.connect(self.findStamps)
        self.ui.btnAnalyzeColors.clicked.connect(self.analyzeColors)
        self.ui.btnScanImage.clicked.connect(self.scanImage)

        self.colorTransform = colorTransform()
        self.cropped_regions = []
        self.colors_detected = []
        self.image_cv_original_rgb = None

        self.settings = QSettings("config.ini", QSettings.IniFormat)
        self.showMaximized()

    def scanImage(self):

        resolution = self.ui.spnDPI.value()

        sane.init()
        devices = sane.get_devices(localOnly=True)
        print(devices[0][0])
        device_name = devices[0][0]

        with sane.open(device_name) as device:
            # print(device.opt)
            device.mode = "Color"
            device.resolution = resolution
            image_pil = self.applyICCTransform(device.scan())
            self.processAndDisplayPilImage(image_pil)
            device.close()
        sane.exit()

    def applyICCTransform(self, image_pil):
        fn_icc_input = self.settings.value("ColorTransform/fileNameInputProfile", None)
        fn_icc_output = self.settings.value("ColorTransform/fileNameOutputProfile", None)
        if fn_icc_input and fn_icc_output:
            return ImageCms.applyTransform(image_pil, ImageCms.ImageCmsTransform(ImageCms.getOpenProfile(fn_icc_input),\
                                                                            ImageCms.getOpenProfile(fn_icc_output),\
                                            input_mode=self.settings.value("ColorTransform/inputColorSystem", "RGB"),\
                                            output_mode=self.settings.value("ColorTransform/outputColorSystem", "RGB")))
        else:
            return image_pil

    def loadImageFromFile(self):
        fileOpenPath = self.settings.value("General/fileOpenPath", ".")
        fileName = QFileDialog.getOpenFileName(self, "Open Image", fileOpenPath , "Image Files (*.png *.jpg *.jpeg *.bmp *tif *tiff)")
        if not fileName[0]:
            return
        self.settings.setValue("General/fileOpenPath", fileName[0])

        image_pil = Image.open(fileName[0]).convert('RGB')
        image_pil = self.applyICCTransform(image_pil)
        self.processAndDisplayPilImage(image_pil)

    def processAndDisplayPilImage(self, image_pil):
        image_cv_original_bgr = cv.cvtColor(np.array(image_pil), cv.COLOR_RGB2BGR)
        self.image_cv_original_rgb = cv.cvtColor(image_cv_original_bgr, cv.COLOR_BGR2RGB)
        self.displayImage(self.image_cv_original_rgb)

    def displayImage(self, image_cv):
        h, w, ch = image_cv.shape
        bytes_per_line = ch * w
        image_qt = QImage(image_cv.data, w, h, bytes_per_line, QImage.Format_RGB888)

        imagePixmap = QPixmap.fromImage(image_qt).scaled(self.ui.lblImageDisplay.width(), self.ui.lblImageDisplay.height(), Qt.KeepAspectRatio)
        #imagePixmap.scaled(self.ui.lblImageDisplay.width(), self.ui.lblImageDisplay.height(), Qt.KeepAspectRatio)
        self.ui.lblImageDisplay.setPixmap(imagePixmap)

    def findStamps(self):
        image_dpi = self.ui.spnDPI.value()

        kernel_size = int(image_dpi/10)
        kernel_size += 1 if kernel_size % 2 == 0 else 0

        # open image and pre-processing
        img = cv.cvtColor(self.image_cv_original_rgb, cv.COLOR_RGB2GRAY)
        img = cv.blur(img, (kernel_size,kernel_size)) # DPI-dependent
        ret,img = cv.threshold(img, self.ui.sldThreshold.value(), 255, cv.THRESH_BINARY)
        img = unsharp_mask(img, (kernel_size,kernel_size))

        # find contours
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernel_size,kernel_size))
        img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel, iterations=1)
        cnts = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        self.cropped_regions = []
        original_rgb = self.image_cv_original_rgb.copy()
        stamp_index = 1
        for c in cnts:
            feature_area = cv.contourArea(c)
            rect = cv.minAreaRect(c)
            center_points = rect[0]
            rect_width, rect_height = rect[1]
            angle = rect[2]
            rect_area = rect_width*rect_height
            if feature_area < image_dpi*2 or rect_width > 5*rect_height or rect_height > 5*rect_width:
                # rough filtering out of ill-detection regions: too small or too non-rect
                continue

            area_ratio = rect_area/feature_area
            delta = min(rect_width, rect_height) / 5
            new_rect_tuple = (center_points, (rect_width/area_ratio-delta, rect_height/area_ratio-delta), angle)
            new_rect = cv.RotatedRect(*new_rect_tuple)
            self.cropped_regions.append( crop_rect(original_rgb, new_rect_tuple) )
            box_points = np.intp(cv.boxPoints(new_rect))
            # draw rectangles and text onto original image
            cv.drawContours(original_rgb, [box_points], 0, (255,255,0), 2)
            original_rgb = cv.putText(original_rgb, str(stamp_index), [int(p) for p in center_points],\
                                      cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 4, cv.LINE_AA)
            original_rgb = cv.putText(original_rgb, str(stamp_index), [int(p) for p in center_points],\
                                      cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2, cv.LINE_AA)
            stamp_index += 1


        self.displayImage(original_rgb)

    def analyzeColors(self):
        if len(self.cropped_regions) > 0:
            self.colors_detected = []
            num_colors = self.ui.spnNumColors.value()

            for region in self.cropped_regions:
                region_colors = []
                colors = self.colorTransform.cluster(region, num_colors)
                colors_lightness = [c["cielab"][0] for c in colors]
                keys_sorted = sorted(range(len(colors_lightness)), key=lambda k: colors_lightness[k])
                for i,color in enumerate(colors):
                    region_colors.append(colors[keys_sorted[i]])
                self.colors_detected.append(region_colors)

            self.populateColorTable()

        # for color in self.colorTransform.cluster(self.selectionLabel.pixmap(), num_colors):
        #     print(f"RGB: {color['rgb']}\nCIEXYZ (D65, 2°): {color['ciexyz']}\nCIELAB (D65, 2°): {color['cielab']}\n" +
        #                 f"\nMatch: Stanley-Gibbons: {color['colorkey']}\nMatch: Munsell: {color['munsell']}\n\n")

    def populateColorTable(self):
        if len(self.colors_detected) > 0:
            self.ui.tblColors.setRowCount(len(self.colors_detected))
            self.ui.tblColors.setColumnCount(len(self.colors_detected[0]))

            for row_index, colors in enumerate(self.colors_detected):
                for col_index, color in enumerate(colors):
                    color_item = QTableWidgetItem(color['munsell'])
                    color_item.setToolTip(f"RGB: {color['rgb']}\nCIEXYZ (D65, 2°): {color['ciexyz']}\nCIELAB (D65, 2°): {color['cielab']}" +
                                          f"\nMatch: Stanley-Gibbons: {color['colorkey']}\nMatch: Munsell: {color['munsell']}")
                    color_item.setBackground(QColor.fromRgb(int(color['rgb_list'][0]), int(color['rgb_list'][1]), int(color['rgb_list'][2])))
                    self.ui.tblColors.setItem(row_index, col_index, color_item)

            self.ui.tblColors.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = Widget()
    widget.show()
    sys.exit(app.exec())
