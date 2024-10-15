from PyQt5.QtGui import QImage
from PyQt5.QtCore import Qt, QSettings, QDir, pyqtSignal, QByteArray, QBuffer, QIODevice

import numpy as np
import pandas as pd
import cv2
from sklearn.cluster import KMeans
import colour


class colorTransform:
    def __init__(self):
        self.colorNames = pd.read_csv("./color_systems/sRGBToMunsellAndISCCNBS.csv", comment="#", delimiter=",")
        self.colorKey = pd.read_csv("./color_systems/color_key_calibrated.csv")
        self.illuminant = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["C"]

    def findRGB(self, color_rgb):
        srgb_rounded5 = np.array([int(5*round(c/5)) for c in color_rgb])
        df = self.colorNames.loc[(self.colorNames["R"] == srgb_rounded5[0]) & (self.colorNames["G"] == srgb_rounded5[1]) & (self.colorNames["B"] == srgb_rounded5[2])]
        if not df.empty:
            return df["Munsell"].loc[df.index[0]], df["ISCC-NBS"].loc[df.index[0]]
        else:
            return "", ""

    def findColorKey(self, color_rgb):
        self.colorKey["distance"] = ((self.colorKey["RGB_R"] - color_rgb[0])**2 + (self.colorKey["RGB_G"] - color_rgb[1])**2 + (self.colorKey["RGB_B"] - color_rgb[2])**2)**0.5
        df = self.colorKey.loc[self.colorKey["distance"]==self.colorKey["distance"].min()]
        return (df["ColorName"].values[0], (df["RGB_R"].values[0], df["RGB_G"].values[0], df["RGB_B"].values[0]))

    def cluster(self, pixmap, num_clusters):
        qimg = pixmap.toImage().convertToFormat(QImage.Format.Format_RGB32)
        ba = QByteArray()
        buff = QBuffer(ba)
        buff.open(QIODevice.ReadWrite) # Essentially open up a "RAM" file
        qimg.save(buff, "PNG") # Store a PNG formatted file into the "RAM" File
        fBytes = np.asarray(bytearray(ba.data()), dtype=np.uint8) # Convert the now PNG contents into a numpy array of bytes
        cropped_region = cv2.imdecode(fBytes, cv2.IMREAD_COLOR) # Let OpenCV "decode" the bytes in RAM as a PNG
        cropped_region = cropped_region.reshape((cropped_region.shape[0] * cropped_region.shape[1], 3))

        clt = KMeans(n_clusters = num_clusters)
        clt.fit(cropped_region)
        colors_rgb = [list(reversed(c)) for c in clt.cluster_centers_]

        matches = []
        for color_rgb in colors_rgb:
            color_xyz = self.rgb2xyz(color_rgb)
            munsell, isccbns = self.findRGB(color_rgb)
            key_name, key_rgb = self.findColorKey(color_rgb)
            matches.append( {"rgb": ", ".join([str(int(c)) for c in color_rgb]),
                             "rgb_list": color_rgb,
                             "cielab": self.xyz2lab(color_xyz),
                             "ciexyz": color_xyz,
                             "munsell": munsell + " (" + isccbns + ")",
                             "colorkey": key_name + " (" + ", ".join([str(int(c)) for c in key_rgb]) + ")",
                             } )
        return matches

    def rgb2xyz(self, RGB, illuminant="D65"):
        # 2 degree observer:
        if illuminant == "D50":
            ill = [96.4212, 100.0, 82.5188]
        else: # D65 and default
            ill = [95.047, 100.0, 108.883]

        RGB = [float(x)/255 for x in RGB]
        RGB = [((x+0.055)/1.055)**2.4 if x > 0.04045 else x/12.92 for x in RGB]
        RGB = [x*100 for x in RGB]

        XYZ = [RGB [0] * 0.4124 + RGB [1] * 0.3576 + RGB [2] * 0.1805,
               RGB [0] * 0.2126 + RGB [1] * 0.7152 + RGB [2] * 0.0722,
               RGB [0] * 0.0193 + RGB [1] * 0.1192 + RGB [2] * 0.9505] # this could/should be a matrix multiplication
        XYZ = [round(x, 4)/ill for x,ill in zip(XYZ,ill)]
        XYZ = [x**0.3333 if x > 0.008856 else (7.787*x)+(16/116) for x in XYZ]

        return [round(x, 4) for x in XYZ]

    def xyz2lab(self, XYZ):
        return [round(x, 4) for x in [(116 * XYZ[1] ) - 16,
                                    500 * ( XYZ[0] - XYZ[1]),
                                    200 * ( XYZ[1] - XYZ[2])]]

    def xyz2xyy(self, XYZ):
        return [round(x, 4) for x in [XYZ[0]/sum(XYZ),
                                    XYZ[1]/sum(XYZ),
                                    XYZ[1]]]

    def xyy2munsell(self, xyY):
        try:
            munsell = colour.xyY_to_munsell_colour(xyY)
        except:
            # when color is outside the Munsell gamut, the routine throws an exception. How convenient :(
            munsell = ""
        return munsell

    def munsellToISCCNBS(self, chroma, value, hue):
        Level3Index = 0

        # Handle the greys first, because they have no hues attached
        if chroma <= 0.5:
            if value >= 8.5:
                Level3Index = 263
            elif value >= 6.5:
                Level3Index = 264
            elif value >= 4.5:
                Level3Index = 265
            elif value >= 2.5:
                Level3Index = 266
            else:
                Level3Index = 267

        else:	# The input colour is not a grey.

            if hue >= 1 and hue < 4:
                if chroma > 0.5 and chroma <= 1.5 and value >= 8.5:
                    Level3Index = 9
                elif chroma > 0.5 and chroma <= 1.5 and value >= 6.5:
                    Level3Index = 10
                elif chroma > 0.5 and chroma <= 1.5 and value >= 4.5:
                    Level3Index = 22
                elif chroma > 0.5 and chroma <= 1.5 and value >= 2.5:
                    Level3Index = 23
                elif chroma > 0.5 and chroma <= 3 and value >= 2 and value < 3.5:
                    Level3Index = 20
                elif chroma > 0.5 and chroma <= 1 and value < 2:
                    Level3Index = 24
                elif chroma > 1 and chroma <= 2 and value < 2:
                    Level3Index = 21
                elif chroma > 1.5 and chroma <= 3 and value >= 8:
                    Level3Index = 7
                elif chroma > 1.5 and chroma <= 3 and value >= 6.5:
                    Level3Index = 8
                elif chroma > 3 and chroma <= 7 and value >= 8:
                    Level3Index = 4
                elif chroma > 3 and chroma <= 7 and value >= 6.5:
                    Level3Index = 5
                elif chroma > 7 and chroma <= 11 and value >= 6.5:
                    Level3Index = 2
                elif chroma > 11 and value >= 6.5:
                    Level3Index = 1
                elif chroma > 1.5 and chroma <= 5 and value >= 5.5 and value < 6.5:
                    Level3Index = 18
                elif chroma > 5 and chroma <= 7 and value >= 5.5 and value < 6.5:
                    Level3Index = 6
                elif chroma > 7 and chroma <= 15 and value >= 5.5 and value < 6.5:
                    Level3Index = 3
                elif chroma > 1.5 and chroma <= 7 and value >= 3.5 and value < 5.5:
                    Level3Index = 19
                elif chroma > 7 and chroma <= 11 and value >= 3.5 and value < 5.5:
                    Level3Index = 15
                elif chroma > 11 and chroma <= 13 and value >= 3.5 and value < 5.5:
                    Level3Index = 12
                elif chroma >= 3 and chroma <= 9 and value >= 2 and value < 3.5:
                    Level3Index = 16
                elif chroma > 9 and chroma <= 11 and value >= 2 and value < 3.5:
                    Level3Index = 13
                elif chroma > 2 and chroma <= 7 and value < 2:
                    Level3Index = 17
                elif chroma > 7 and chroma <= 11 and value < 2:
                    Level3Index = 14
                elif chroma > 11 and value < 6.5:
                    Level3Index = 11
                else:
                    print("no match")

            elif hue >= 4 and hue < 6:
                if chroma > 0.5 and chroma <= 1.5 and value >= 8.5:
                    Level3Index = 9
                elif chroma > 0.5 and chroma <= 1.5 and value >= 6.5:
                    Level3Index = 10
                elif chroma > 0.5 and chroma <= 1.5 and value >= 4.5:
                    Level3Index = 22
                elif chroma > 0.5 and chroma <= 1.5 and value >= 2.5:
                    Level3Index = 23
                elif chroma > 0.5 and chroma <= 3 and value >= 2 and value < 3.5:
                    Level3Index = 20
                elif chroma > 0.5 and chroma <= 1 and value < 2:
                    Level3Index = 24
                elif chroma > 1 and chroma <= 2 and value < 2:
                    Level3Index = 21
                elif chroma > 1.5 and chroma <= 3 and value >= 8:
                    Level3Index = 7
                elif chroma > 1.5 and chroma <= 3 and value >= 6.5:
                    Level3Index = 8
                elif chroma > 3 and chroma <= 7 and value >= 8:
                    Level3Index = 4
                elif chroma > 3 and chroma <= 7 and value >= 6.5:
                    Level3Index = 5
                elif chroma > 7 and chroma <= 11 and value >= 6.5:
                    Level3Index = 26
                elif chroma > 11 and value >= 6.5:
                    Level3Index = 25
                elif chroma > 1.5 and chroma <= 5 and value >= 5.5 and value < 6.5:
                    Level3Index = 18
                elif chroma > 5 and chroma <= 7 and value >= 5.5 and value < 6.5:
                    Level3Index = 6
                elif chroma > 7 and chroma <= 11 and value >= 5.5 and value < 6.5:
                    Level3Index = 3
                elif chroma > 11 and chroma <= 15 and value >= 5.5 and value < 6.5:
                    Level3Index = 27
                elif chroma > 1.5 and chroma <= 7 and value >= 3.5 and value < 5.5:
                    Level3Index = 19
                elif chroma > 7 and chroma <= 11 and value >= 3.5 and value < 5.5:
                    Level3Index = 15
                elif chroma > 11 and chroma <= 13 and value >= 3.5 and value < 5.5:
                    Level3Index = 12
                elif chroma >= 3 and chroma <= 9 and value >= 2 and value < 3.5:
                    Level3Index = 16
                elif chroma > 9 and chroma <= 11 and value >= 2 and value < 3.5:
                    Level3Index = 13
                elif chroma > 2 and chroma <= 7 and value < 2:
                    Level3Index = 17
                elif chroma > 7 and chroma <= 11 and value < 2:
                    Level3Index = 14
                elif chroma > 11 and value < 6.5:
                    Level3Index = 11
                else:
                    print("no match")

            elif hue >= 6 and hue < 7:
                if chroma > 0.5 and chroma <= 1.5 and value >= 8.5:
                    Level3Index = 9
                elif chroma > 0.5 and chroma <= 1.5 and value >= 6.5:
                    Level3Index = 10
                elif chroma > 0.5 and chroma <= 1.5 and value >= 4.5:
                    Level3Index = 22
                elif chroma > 0.5 and chroma <= 1.5 and value >= 2.5:
                    Level3Index = 23
                elif chroma > 0.5 and chroma <= 3 and value >= 1.5 and value < 2.5:
                    Level3Index = 47
                elif chroma > 1.5 and chroma <= 3 and value >= 2.5 and value < 3.5:
                    Level3Index = 46
                elif chroma > 0.5 and chroma <= 1 and value < 1.5:
                    Level3Index = 24
                elif chroma > 1 and chroma <= 5 and value < 2.5:
                    Level3Index = 44
                elif chroma > 3 and chroma <= 7 and value >= 2.5 and value < 3.5:
                    Level3Index = 43
                elif chroma > 5 and chroma <= 7 and value < 2.5:
                    Level3Index = 41
                elif chroma > 1.5 and chroma <= 3 and value >= 8:
                    Level3Index = 31
                elif chroma > 1.5 and chroma <= 3 and value >= 6.5:
                    Level3Index = 32
                elif chroma > 3 and chroma <= 7 and value >= 8:
                    Level3Index = 28
                elif chroma > 3 and chroma <= 7 and value >= 6.5:
                    Level3Index = 29
                elif chroma > 7 and chroma <= 11 and value >= 6.5:
                    Level3Index = 26
                elif chroma > 11 and value >= 6.5:
                    Level3Index = 25
                elif chroma > 1.5 and chroma <= 5 and value >= 5.5 and value < 6.5:
                    Level3Index = 18
                elif chroma > 5 and chroma <= 7 and value >= 5.5 and value < 6.5:
                    Level3Index = 30
                elif chroma > 7 and chroma <= 15 and value >= 5.5 and value < 6.5:
                    Level3Index = 27
                elif chroma > 1.5 and chroma <= 7 and value >= 3.5 and value < 5.5:
                    Level3Index = 19
                elif chroma > 7 and chroma <= 11 and value >= 3.5 and value < 5.5:
                    Level3Index = 15
                elif chroma > 11 and chroma <= 13 and value >= 3.5 and value < 5.5:
                    Level3Index = 12
                elif chroma >= 7 and chroma <= 9 and value >= 2 and value < 3.5:
                    Level3Index = 16
                elif chroma > 9 and chroma <= 11 and value >= 2 and value < 3.5:
                    Level3Index = 13
                elif chroma > 7 and chroma <= 11 and value < 2:
                    Level3Index = 14
                elif chroma > 11 and value < 6.5:
                    Level3Index = 11
                else:
                    print("no match")

            elif hue >= 7 and hue < 8:
                if chroma > 0.5 and chroma <= 1.5 and value >= 8.5:
                    Level3Index = 9
                elif chroma > 0.5 and chroma <= 1.5 and value >= 6.5:
                    Level3Index = 10
                elif chroma > 0.5 and chroma <= 1.5 and value >= 4.5:
                    Level3Index = 22
                elif chroma > 0.5 and chroma <= 1.5 and value >= 2.5:
                    Level3Index = 23
                elif chroma > 0.5 and chroma <= 3 and value >= 1.5 and value < 2.5:
                    Level3Index = 47
                elif chroma > 1.5 and chroma <= 3 and value >= 2.5 and value <= 3.5:
                    Level3Index = 46
                elif chroma > 1.5 and chroma <= 7 and value >= 3.5 and value <= 5.5:
                    Level3Index = 19
                elif chroma > 0.5 and chroma <= 1 and value < 1.5:
                    Level3Index = 24
                elif chroma > 1 and chroma <= 5 and value < 2.5:
                    Level3Index = 44
                elif chroma > 3 and chroma <= 7 and value >= 2.5 and value < 3.5:
                    Level3Index = 43
                elif chroma > 5 and chroma <= 7 and value < 2.5:
                    Level3Index = 41
                elif chroma > 1.5 and chroma <= 3 and value >= 8:
                    Level3Index = 31
                elif chroma > 1.5 and chroma <= 3 and value >= 6.5:
                    Level3Index = 32
                elif chroma > 3 and chroma <= 7 and value >= 8:
                    Level3Index = 28
                elif chroma > 3 and chroma <= 7 and value >= 6.5:
                    Level3Index = 29
                elif chroma > 7 and chroma <= 11 and value >= 6.5:
                    Level3Index = 26
                elif chroma > 11 and value >= 6.5:
                    Level3Index = 25
                elif chroma > 1.5 and chroma <= 5 and value >= 5.5 and value < 6.5:
                    Level3Index = 18
                elif chroma > 5 and chroma <= 7 and value >= 5.5 and value < 6.5:
                    Level3Index = 30
                elif chroma > 7 and chroma <= 11 and value >= 4.5 and value < 6.5:
                    Level3Index = 37
                elif chroma > 7 and chroma <= 11 and value >= 3.5 and value < 4.5:
                    Level3Index = 38
                elif chroma > 11 and chroma <= 13 and value >= 4.5 and value < 6.5:
                    Level3Index = 35
                elif chroma > 11 and chroma <= 13 and value >= 3.5 and value < 4.5:
                    Level3Index = 36
                elif chroma > 13 and value >= 4.5 and value < 6.5:
                    Level3Index = 34
                elif chroma >= 7 and chroma <= 9 and value >= 2 and value < 3.5:
                    Level3Index = 16
                elif chroma > 9 and chroma <= 11 and value >= 2 and value < 3.5:
                    Level3Index = 13
                elif chroma > 7 and chroma <= 11 and value < 2:
                    Level3Index = 14
                elif chroma > 11 and value < 4.5:
                    Level3Index = 11
                else:
                    print("no match")

            elif hue >= 8 and hue < 9:
                if chroma > 0.5 and chroma <= 1.5 and value >= 8.5:
                    Level3Index = 9
                elif chroma > 0.5 and chroma <= 1.5 and value >= 6.5:
                    Level3Index = 10
                elif chroma > 0.5 and chroma <= 1.5 and value >= 4.5:
                    Level3Index = 22
                elif chroma > 0.5 and chroma <= 1.5 and value >= 2.5:
                    Level3Index = 23
                elif chroma > 0.5 and chroma <= 3 and value >= 1.5 and value < 2.5:
                    Level3Index = 47
                elif chroma > 1.5 and chroma <= 3 and value >= 2.5 and value < 4.5:
                    Level3Index = 46
                elif chroma > 0.5 and chroma <= 1 and value < 1.5:
                    Level3Index = 24
                elif chroma > 1 and chroma <= 5 and value < 2.5:
                    Level3Index = 44
                elif chroma > 3 and chroma <= 7 and value >= 2.5 and value < 4.5:
                    Level3Index = 43
                elif chroma > 5 and chroma <= 7 and value < 2.5:
                    Level3Index = 41
                elif chroma > 1.5 and chroma <= 3 and value >= 8:
                    Level3Index = 31
                elif chroma > 1.5 and chroma <= 3 and value >= 6.5:
                    Level3Index = 32
                elif chroma > 3 and chroma <= 7 and value >= 8:
                    Level3Index = 28
                elif chroma > 3 and chroma <= 7 and value >= 6.5:
                    Level3Index = 29
                elif chroma > 7 and chroma <= 11 and value >= 6.5:
                    Level3Index = 26
                elif chroma > 11 and value >= 6.5:
                    Level3Index = 25
                elif chroma > 1.5 and chroma <= 3 and value >= 5.5 and value < 6.5:
                    Level3Index = 18
                elif chroma > 1.5 and chroma <= 3 and value >= 4.5 and value < 5.5:
                    Level3Index = 19
                elif chroma > 3 and chroma <= 5 and value >= 4.5 and value < 6.5:
                    Level3Index = 42
                elif chroma > 5 and chroma <= 7 and value >= 4.5 and value < 6.5:
                    Level3Index = 39
                elif chroma > 7 and chroma <= 11 and value >= 4.5 and value < 6.5:
                    Level3Index = 37
                elif chroma > 7 and chroma <= 11 and value >= 3.5 and value < 4.5:
                    Level3Index = 38
                elif chroma > 11 and chroma <= 13 and value >= 4.5 and value < 6.5:
                    Level3Index = 35
                elif chroma > 11 and chroma <= 13 and value >= 3.5 and value < 4.5:
                    Level3Index = 36
                elif chroma > 13 and value >= 4.5 and value < 6.5:
                    Level3Index = 34
                elif chroma >= 7 and chroma <= 9 and value >= 2 and value < 3.5:
                    Level3Index = 16
                elif chroma > 9 and chroma <= 11 and value >= 2 and value < 3.5:
                    Level3Index = 13
                elif chroma > 7 and chroma <= 11 and value < 2:
                    Level3Index = 14
                elif chroma > 11 and value < 4.5:
                    Level3Index = 11
                else:
                    print("no match")

            elif hue >= 9 and hue < 11:
                if chroma > 0.5 and chroma <= 1.5 and value >= 8.5:
                    Level3Index = 9
                elif chroma > 0.5 and chroma <= 1.5 and value >= 6.5:
                    Level3Index = 10
                elif chroma > 0.5 and chroma <= 1.5 and value >= 4.5:
                    Level3Index = 22
                elif chroma > 0.5 and chroma <= 1.5 and value >= 2.5:
                    Level3Index = 23
                elif chroma > 0.5 and chroma <= 3 and value >= 1.5 and value < 2.5:
                    Level3Index = 47
                elif chroma > 1.5 and chroma <= 3 and value >= 2.5 and value < 4.5:
                    Level3Index = 46
                elif chroma > 0.5 and chroma <= 1 and value < 1.5:
                    Level3Index = 24
                elif chroma > 1 and chroma <= 5 and value < 2.5:
                    Level3Index = 44
                elif chroma > 3 and chroma <= 7 and value >= 2.5 and value < 4.5:
                    Level3Index = 43
                elif chroma > 5 and value < 2.5:
                    Level3Index = 41
                elif chroma > 1.5 and chroma <= 3 and value >= 8:
                    Level3Index = 31
                elif chroma > 1.5 and chroma <= 3 and value >= 6.5:
                    Level3Index = 32
                elif chroma > 3 and chroma <= 7 and value >= 8:
                    Level3Index = 28
                elif chroma > 3 and chroma <= 7 and value >= 6.5:
                    Level3Index = 29
                elif chroma > 7 and chroma <= 11 and value >= 6.5:
                    Level3Index = 26
                elif chroma > 11 and value >= 6.5:
                    Level3Index = 25
                elif chroma > 1.5 and chroma <= 3 and value >= 5.5 and value < 6.5:
                    Level3Index = 18
                elif chroma > 1.5 and chroma <= 3 and value >= 4.5 and value < 5.5:
                    Level3Index = 19
                elif chroma > 3 and chroma <= 5 and value >= 4.5 and value < 6.5:
                    Level3Index = 42
                elif chroma > 5 and chroma <= 7 and value >= 4.5 and value < 6.5:
                    Level3Index = 39
                elif chroma > 7 and chroma <= 11 and value >= 4.5 and value < 6.5:
                    Level3Index = 37
                elif chroma > 7 and chroma <= 11 and value >= 3.5 and value < 4.5:
                    Level3Index = 38
                elif chroma > 11 and chroma <= 13 and value >= 4.5 and value < 6.5:
                    Level3Index = 35
                elif chroma > 11 and chroma <= 13 and value >= 3.5 and value < 4.5:
                    Level3Index = 36
                elif chroma > 13 and value >= 3.5 and value < 6.5:
                    Level3Index = 34
                elif chroma >= 7 and value >= 2.5 and value < 3.5:
                    Level3Index = 40
                else:
                    print("no match")

            elif hue >= 11 and hue < 12:
                if chroma > 0.5 and chroma <= 1.5 and value >= 8.5:
                    Level3Index = 9
                elif chroma > 0.5 and chroma <= 1.5 and value >= 6.5:
                    Level3Index = 10
                elif chroma > 0.5 and chroma <= 1.5 and value >= 4.5:
                    Level3Index = 22
                elif chroma > 0.5 and chroma <= 1.5 and value >= 2.5:
                    Level3Index = 64
                elif chroma > 0.5 and chroma <= 3 and value >= 1.5 and value < 2.5:
                    Level3Index = 47
                elif chroma > 1.5 and chroma <= 3 and value >= 2.5 and value < 4.5:
                    Level3Index = 46
                elif chroma > 0.5 and chroma <= 1 and value < 1.5:
                    Level3Index = 65
                elif chroma > 1 and chroma <= 5 and value < 2.5:
                    Level3Index = 44
                elif chroma > 3 and chroma <= 7 and value >= 2.5 and value < 4.5:
                    Level3Index = 43
                elif chroma > 5 and value < 2.5:
                    Level3Index = 41
                elif chroma > 1.5 and chroma <= 3 and value >= 8:
                    Level3Index = 31
                elif chroma > 1.5 and chroma <= 3 and value >= 6.5:
                    Level3Index = 32
                elif chroma > 3 and chroma <= 7 and value >= 8:
                    Level3Index = 28
                elif chroma > 3 and chroma <= 7 and value >= 6.5:
                    Level3Index = 29
                elif chroma > 7 and chroma <= 11 and value >= 6.5:
                    Level3Index = 26
                elif chroma > 11 and value >= 6.5:
                    Level3Index = 25
                elif chroma > 1.5 and chroma <= 3 and value >= 4.5 and value < 6.5:
                    Level3Index = 45
                elif chroma > 3 and chroma <= 5 and value >= 4.5 and value < 6.5:
                    Level3Index = 42
                elif chroma > 5 and chroma <= 7 and value >= 4.5 and value < 6.5:
                    Level3Index = 39
                elif chroma > 7 and chroma <= 11 and value >= 4.5 and value < 6.5:
                    Level3Index = 37
                elif chroma > 7 and chroma <= 11 and value >= 3.5 and value < 4.5:
                    Level3Index = 38
                elif chroma > 11 and chroma <= 13 and value >= 4.5 and value < 6.5:
                    Level3Index = 35
                elif chroma > 11 and chroma <= 13 and value >= 3.5 and value < 4.5:
                    Level3Index = 36
                elif chroma > 13 and value >= 3.5 and value < 6.5:
                    Level3Index = 34
                elif chroma >= 7 and value >= 2.5 and value < 3.5:
                    Level3Index = 40
                else:
                    print("no match")

            elif hue >= 12 and hue < 13:
                if chroma > 0.5 and chroma <= 1.5 and value >= 8.5:
                    Level3Index = 9
                elif chroma > 0.5 and chroma <= 1.5 and value >= 6.5:
                    Level3Index = 10
                elif chroma > 0.5 and chroma <= 1.5 and value >= 4.5:
                    Level3Index = 63
                elif chroma > 0.5 and chroma <= 1.5 and value >= 2.5:
                    Level3Index = 64
                elif chroma > 0.5 and chroma <= 3 and value >= 1.5 and value < 2.5:
                    Level3Index = 47
                elif chroma > 1.5 and chroma <= 3 and value >= 2.5 and value < 4.5:
                    Level3Index = 46
                elif chroma > 0.5 and chroma <= 1 and value < 1.5:
                    Level3Index = 65
                elif chroma > 1 and chroma <= 5 and value < 2.5:
                    Level3Index = 44
                elif chroma > 3 and chroma <= 5 and value >= 2.5 and value < 4.5:
                    Level3Index = 43
                elif chroma > 5 and value < 2.5:
                    Level3Index = 56
                elif chroma > 1.5 and chroma <= 3 and value >= 8:
                    Level3Index = 31
                elif chroma > 1.5 and chroma <= 3 and value >= 6.5:
                    Level3Index = 32
                elif chroma > 1.5 and chroma <= 3 and value >= 4.5 and value < 6.5:
                    Level3Index = 45
                elif chroma > 1.5 and chroma <= 3 and value >= 2.5 and value < 4.5:
                    Level3Index = 46
                elif chroma > 3 and chroma <= 6 and value >= 8:
                    Level3Index = 28
                elif chroma > 3 and chroma <= 6 and value >= 6.5:
                    Level3Index = 29
                elif chroma > 6 and chroma <= 10 and value >= 7.5:
                    Level3Index = 52
                elif chroma > 3 and chroma <= 5 and value >= 4.5 and value < 6.5:
                    Level3Index = 42
                elif chroma > 5 and chroma <= 7 and value >= 4.5 and value < 6.5:
                    Level3Index = 39
                elif chroma > 6 and chroma <= 10 and value >= 5.5:
                    Level3Index = 53
                elif chroma > 7 and chroma <= 10 and value >= 4.5:
                    Level3Index = 54
                elif chroma > 10 and chroma <= 14 and value >= 7.5:
                    Level3Index = 49
                elif chroma > 10 and chroma <= 14 and value >= 5.5:
                    Level3Index = 50
                elif chroma > 10 and chroma <= 14 and value >= 4.5:
                    Level3Index = 51
                elif chroma > 14 and value >= 4.5:
                    Level3Index = 48
                elif chroma > 5 and value >= 2.5 and value < 4.5:
                    Level3Index = 55
                else:
                    print("no match")

            elif hue >= 13 and hue < 15:
                if chroma > 0.5 and chroma <= 1.5 and value >= 8.5:
                    Level3Index = 9
                elif chroma > 0.5 and chroma <= 1.5 and value >= 6.5:
                    Level3Index = 10
                elif chroma > 0.5 and chroma <= 1.5 and value >= 4.5:
                    Level3Index = 63
                elif chroma > 0.5 and chroma <= 1.5 and value >= 2.5:
                    Level3Index = 64
                elif chroma > 1.5 and chroma <= 2.5 and value >= 2.5 and value < 4.5:
                    Level3Index = 61
                elif chroma > 0.5 and chroma <= 1 and value < 1.5:
                    Level3Index = 65
                elif chroma > 0.5 and chroma <= 2.5 and value >= 1.5 and value < 2.5:
                    Level3Index = 62
                elif chroma > 1 and chroma <= 5 and value < 2.5:
                    Level3Index = 59
                elif chroma > 2.5 and chroma <= 5 and value >= 2.5 and value < 4.5:
                    Level3Index = 58
                elif chroma > 5 and value < 2.5:
                    Level3Index = 56
                elif chroma > 1.5 and chroma <= 3 and value >= 8:
                    Level3Index = 31
                elif chroma > 1.5 and chroma <= 3 and value >= 6.5:
                    Level3Index = 32
                elif chroma > 1.5 and chroma <= 3 and value >= 4.5 and value < 6.5:
                    Level3Index = 45
                elif chroma > 3 and chroma <= 6 and value >= 8:
                    Level3Index = 28
                elif chroma > 3 and chroma <= 6 and value >= 6.5:
                    Level3Index = 29
                elif chroma > 3 and chroma <= 6 and value >= 4.5 and value < 6.5:
                    Level3Index = 57
                elif chroma > 6 and chroma <= 10 and value >= 7.5:
                    Level3Index = 52
                elif chroma > 6 and chroma <= 10 and value >= 5.5:
                    Level3Index = 53
                elif chroma > 6 and chroma <= 10 and value >= 4.5:
                    Level3Index = 54
                elif chroma > 10 and chroma <= 14 and value >= 7.5:
                    Level3Index = 49
                elif chroma > 10 and chroma <= 14 and value >= 5.5:
                    Level3Index = 50
                elif chroma > 10 and chroma <= 14 and value >= 4.5:
                    Level3Index = 51
                elif chroma > 14 and value >= 4.5:
                    Level3Index = 48
                elif chroma > 5 and value >= 2.5 and value < 4.5:
                    Level3Index = 55
                else:
                    print("no match")

            elif hue >= 15 and hue < 17:
                if chroma > 0.5 and chroma <= 1.2 and value >= 8.5:
                    Level3Index = 9
                elif chroma > 0.5 and chroma <= 1.2 and value >= 6.5:
                    Level3Index = 10
                elif chroma > 0.5 and chroma <= 1.2 and value >= 4.5:
                    Level3Index = 63
                elif chroma > 0.5 and chroma <= 1.2 and value >= 2.5:
                    Level3Index = 64
                elif chroma > 1.2 and chroma <= 2.5 and value >= 2.5 and value < 4.5:
                    Level3Index = 61
                elif chroma > 0.5 and chroma <= 1 and value < 1.5:
                    Level3Index = 65
                elif chroma > 0.5 and chroma <= 2.5 and value >= 1.5 and value < 2.5:
                    Level3Index = 62
                elif chroma > 1 and chroma <= 5 and value < 2.5:
                    Level3Index = 59
                elif chroma > 2.5 and chroma <= 5 and value >= 2.5 and value < 4.5:
                    Level3Index = 58
                elif chroma > 5 and value < 2.5:
                    Level3Index = 56
                elif chroma > 1.2 and chroma <= 3 and value >= 8:
                    Level3Index = 31
                elif chroma > 1.2 and chroma <= 3 and value >= 6.5:
                    Level3Index = 33
                elif chroma > 1.2 and chroma <= 3 and value >= 4.5 and value < 6.5:
                    Level3Index = 60
                elif chroma > 3 and chroma <= 6 and value >= 8:
                    Level3Index = 28
                elif chroma > 3 and chroma <= 6 and value >= 6.5:
                    Level3Index = 29
                elif chroma > 3 and chroma <= 6 and value >= 4.5 and value < 6.5:
                    Level3Index = 57
                elif chroma > 6 and chroma <= 10 and value >= 7.5:
                    Level3Index = 52
                elif chroma > 6 and chroma <= 10 and value >= 5.5:
                    Level3Index = 53
                elif chroma > 6 and chroma <= 10 and value >= 4.5:
                    Level3Index = 54
                elif chroma > 10 and chroma <= 14 and value >= 7.5:
                    Level3Index = 49
                elif chroma > 10 and chroma <= 14 and value >= 5.5:
                    Level3Index = 50
                elif chroma > 10 and chroma <= 14 and value >= 4.5:
                    Level3Index = 51
                elif chroma > 14 and value >= 4.5:
                    Level3Index = 48
                elif chroma > 5 and value >= 2.5 and value < 4.5:
                    Level3Index = 55
                else:
                    print("no match")

            elif hue >= 17 and hue < 18:
                if chroma > 0.7 and chroma <= 1.2 and value >= 8.5:
                    Level3Index = 92
                elif chroma > 0.7 and chroma <= 1.2 and value >= 6.5:
                    Level3Index = 93
                elif chroma > 0.7 and chroma <= 1.2 and value >= 4.5:
                    Level3Index = 63
                elif value >= 8.5 and chroma <= 0.7:
                    Level3Index = 263
                elif value >= 6.5 and chroma <= 0.7:
                    Level3Index = 264
                elif value >= 4.5 and chroma <= 0.7:
                    Level3Index = 265
                elif chroma > 0.5 and chroma <= 1.2 and value >= 2.5 and value < 4.5:
                    Level3Index = 64
                elif chroma > 1.2 and chroma <= 2.5 and value >= 2.5 and value < 4.5:
                    Level3Index = 61
                elif chroma > 0.5 and chroma <= 1 and value < 1.5:
                    Level3Index = 65
                elif chroma > 0.5 and chroma <= 2.5 and value >= 1.5 and value < 2.5:
                    Level3Index = 62
                elif chroma > 1 and chroma <= 5 and value < 2.5:
                    Level3Index = 59
                elif chroma > 2.5 and chroma <= 5 and value >= 2.5 and value < 4.5:
                    Level3Index = 58
                elif chroma > 1.2 and chroma <= 3 and value >= 8:
                    Level3Index = 31
                elif chroma > 1.2 and chroma <= 3 and value >= 6.5:
                    Level3Index = 33
                elif chroma > 1.2 and chroma <= 3 and value >= 4.5 and value < 6.5:
                    Level3Index = 60
                elif chroma > 3 and chroma <= 6 and value >= 7.5:
                    Level3Index = 73
                elif chroma > 3 and chroma <= 6 and value >= 6.5:
                    Level3Index = 76
                elif chroma > 3 and chroma <= 6 and value >= 4.5 and value < 6.5:
                    Level3Index = 57
                elif chroma > 6 and chroma <= 10 and value >= 8:
                    Level3Index = 70
                elif chroma > 6 and chroma <= 10 and value >= 6.5:
                    Level3Index = 71
                elif chroma > 6 and chroma <= 10 and value >= 5.5:
                    Level3Index = 72
                elif chroma > 10 and chroma <= 14 and value >= 8:
                    Level3Index = 67
                elif chroma > 10 and chroma <= 14 and value >= 6.5:
                    Level3Index = 68
                elif chroma > 10 and chroma <= 14 and value >= 5.5:
                    Level3Index = 69
                elif chroma > 6 and value >= 4.5 and value < 5.5:
                    Level3Index = 74
                elif chroma > 14 and value >= 5.5:
                    Level3Index = 66
                elif chroma > 5 and value >= 2.5 and value < 4.5:
                    Level3Index = 55
                elif chroma > 5 and value < 2.5:
                    Level3Index = 56
                else:
                    print("no match")

            elif hue >= 18 and hue < 21:
                if chroma > 0.7 and chroma <= 2 and value >= 8.5:
                    Level3Index = 92
                elif chroma > 0.7 and chroma <= 2 and value >= 6.5:
                    Level3Index = 93
                elif chroma > 0.7 and chroma <= 1.2 and value >= 4.5 and value < 6.5:
                    Level3Index = 63
                elif value >= 8.5 and chroma <= 0.7:
                    Level3Index = 263
                elif value >= 6.5 and chroma <= 0.7:
                    Level3Index = 264
                elif value >= 4.5 and chroma <= 0.7:
                    Level3Index = 265
                elif chroma > 0.5 and chroma <= 1.2 and value >= 2.5 and value < 4.5:
                    Level3Index = 64
                elif chroma > 0.5 and chroma <= 1 and value <= 1.5:
                    Level3Index = 65
                elif chroma > 2 and chroma <= 6 and value >= 7.5:
                    Level3Index = 73
                elif chroma > 1.2 and chroma <= 3 and value >= 5.5 and value < 7.5:
                    Level3Index = 79
                elif chroma > 3 and chroma <= 6 and value >= 5.5 and value < 7.5:
                    Level3Index = 76
                elif (chroma > 1.2 and chroma <= 3 and value >= 4.5 and value < 5.5) or (chroma > 1.2 and chroma <= 2.5 and value >= 3.5 and value < 4.5):
                    Level3Index = 80
                elif (chroma > 3 and chroma <= 5 and value >= 4.5 and value < 5.5) or (chroma > 2.5 and chroma <= 5 and value >= 3.5 and value < 4.5):
                    Level3Index = 77
                elif (chroma > 1.2 and chroma <= 2.5 and value >= 2.5 and value < 3.5) or (chroma > 0.5 and chroma <= 2.5 and value >= 1.5 and value < 2.5):
                    Level3Index = 81
                elif chroma > 1 and chroma <= 5 and value < 3.5:
                    Level3Index = 78
                elif chroma > 6 and chroma <= 10 and value >= 8:
                    Level3Index = 70
                elif chroma > 6 and chroma <= 10 and value >= 6.5:
                    Level3Index = 71
                elif chroma > 6 and chroma <= 10 and value >= 5.5:
                    Level3Index = 72
                elif chroma > 10 and chroma <= 14 and value >= 8:
                    Level3Index = 67
                elif chroma > 10 and chroma <= 14 and value >= 6.5:
                    Level3Index = 68
                elif chroma > 10 and chroma <= 14 and value >= 5.5:
                    Level3Index = 69
                elif chroma > 14 and value >= 5.5:
                    Level3Index = 66
                elif chroma > 5 and value >= 3.5 and value < 5.5:
                    Level3Index = 74
                elif chroma > 5 and value < 3.5:
                    Level3Index = 75
                else:
                    print("no match")

            elif hue >= 21 and hue < 24:
                if chroma > 0.7 and chroma <= 2 and value >= 8.5:
                    Level3Index = 92
                elif chroma > 0.7 and chroma <= 2 and value >= 6.5:
                    Level3Index = 93
                elif chroma > 0.7 and chroma <= 1.2 and value >= 4.5 and value < 6.5:
                    Level3Index = 63
                elif value >= 8.5 and chroma <= 0.7:
                    Level3Index = 263
                elif value >= 6.5 and chroma <= 0.7:
                    Level3Index = 264
                elif value >= 4.5 and chroma <= 0.7:
                    Level3Index = 265
                elif chroma > 0.5 and chroma <= 1.2 and value >= 2.5 and value < 4.5:
                    Level3Index = 64
                elif chroma > 0.5 and chroma <= 1 and value < 1.5:
                    Level3Index = 65
                elif chroma > 2 and chroma <= 5 and value >= 8:
                    Level3Index = 89
                elif chroma > 2 and chroma <= 5 and value >= 6.5:
                    Level3Index = 90
                elif chroma > 3 and chroma <= 5 and value >= 5.5 and value < 6.5:
                    Level3Index = 91
                elif chroma > 5 and chroma <= 8 and value >= 8:
                    Level3Index = 86
                elif chroma > 5 and chroma <= 8 and value >= 6.5:
                    Level3Index = 87
                elif chroma > 5 and chroma <= 8 and value >= 5.5:
                    Level3Index = 88
                elif chroma > 8 and chroma <= 11 and value >= 8:
                    Level3Index = 83
                elif chroma > 8 and chroma <= 11 and value >= 6.5:
                    Level3Index = 84
                elif chroma > 8 and chroma <= 11 and value >= 5.5:
                    Level3Index = 85
                elif chroma > 11 and value >= 5.5:
                    Level3Index = 82
                elif chroma > 1.2 and value >= 4.5 and value < 6.5:
                    Level3Index = 94
                elif chroma > 1.2 and value >= 2.5 and value < 4.5:
                    Level3Index = 95
                elif chroma > 0.5 and value < 2.5:
                    Level3Index = 96
                else:
                    print("no match")

            elif hue >= 24 and hue < 27:
                if chroma > 0.7 and chroma <= 2 and value >= 8.5:
                    Level3Index = 92
                elif chroma > 0.7 and chroma <= 2 and value >= 6.5:
                    Level3Index = 93
                elif chroma > 0.7 and chroma <= 2 and value >= 4.5 and value < 6.5:
                    Level3Index = 112
                elif chroma > 2 and chroma <= 3 and value >= 4.5 and value < 6.5:
                    Level3Index = 109
                elif value >= 8.5 and chroma <= 0.7:
                    Level3Index = 263
                elif value >= 6.5 and chroma <= 0.7:
                    Level3Index = 264
                elif value >= 4.5 and chroma <= 0.7:
                    Level3Index = 265
                elif chroma > 0.5 and chroma <= 1.5 and value >= 2.5 and value < 4.5:
                    Level3Index = 113
                elif chroma > 1.5 and chroma <= 3 and value >= 2.5 and value < 4.5:
                    Level3Index = 110
                elif chroma > 0.5 and chroma <= 3 and value >= 1.5 and value < 2.5:
                    Level3Index = 111
                elif chroma > 0.5 and chroma <= 1 and value < 1.5:
                    Level3Index = 114
                elif chroma > 2 and chroma <= 5 and value >= 8:
                    Level3Index = 89
                elif chroma > 2 and chroma <= 5 and value >= 6.5:
                    Level3Index = 90
                elif chroma > 3 and chroma <= 5 and value >= 5.5 and value < 6.5:
                    Level3Index = 91
                elif chroma > 5 and chroma <= 8 and value >= 8:
                    Level3Index = 86
                elif chroma > 5 and chroma <= 8 and value >= 6.5:
                    Level3Index = 87
                elif chroma > 5 and chroma <= 8 and value >= 5.5:
                    Level3Index = 88
                elif chroma > 8 and chroma <= 11 and value >= 8:
                    Level3Index = 83
                elif chroma > 8 and chroma <= 11 and value >= 6.5:
                    Level3Index = 84
                elif chroma > 8 and chroma <= 11 and value >= 5.5:
                    Level3Index = 85
                elif chroma > 11 and value >= 5.5:
                    Level3Index = 82
                elif chroma > 3 and value >= 4.5 and value < 5.5:
                    Level3Index = 106
                elif chroma > 3 and value >= 2.5 and value < 4.5:
                    Level3Index = 107
                elif chroma > 0.5 and value < 2.5:
                    Level3Index = 108
                else:
                    print("no match")

            elif hue >= 27 and hue < 29:
                if chroma > 0.7 and chroma <= 2 and value >= 8.5:
                    Level3Index = 92
                elif chroma > 0.7 and chroma <= 2 and value >= 6.5:
                    Level3Index = 93
                elif chroma > 0.7 and chroma <= 2 and value >= 4.5 and value < 6.5:
                    Level3Index = 112
                elif chroma > 2 and chroma <= 3 and value >= 8:
                    Level3Index = 89
                elif chroma > 2 and chroma <= 3 and value >= 6.5:
                    Level3Index = 90
                elif chroma > 2 and chroma <= 3 and value >= 4.5 and value < 6.5:
                    Level3Index = 109
                elif value >= 8.5 and chroma <= 0.7:
                    Level3Index = 263
                elif value >= 6.5 and chroma <= 0.7:
                    Level3Index = 264
                elif value >= 4.5 and chroma <= 0.7:
                    Level3Index = 265
                elif chroma > 0.5 and chroma <= 1.5 and value >= 2.5 and value < 4.5:
                    Level3Index = 113
                elif chroma > 1.5 and chroma <= 3 and value >= 2.5 and value < 4.5:
                    Level3Index = 110
                elif chroma > 0.5 and chroma <= 3 and value >= 1.5 and value < 2.5:
                    Level3Index = 111
                elif chroma > 0.5 and chroma <= 1 and value < 1.5:
                    Level3Index = 114
                elif chroma > 3 and chroma <= 5 and value >= 8:
                    Level3Index = 104
                elif chroma > 3 and chroma <= 5 and value >= 6.5:
                    Level3Index = 105
                elif chroma > 5 and chroma <= 8 and value >= 8:
                    Level3Index = 101
                elif chroma > 5 and chroma <= 8 and value >= 6.5:
                    Level3Index = 102
                elif chroma > 5 and chroma <= 8 and value >= 5.5:
                    Level3Index = 103
                elif chroma > 8 and chroma <= 11 and value >= 8:
                    Level3Index = 98
                elif chroma > 8 and chroma <= 11 and value >= 6.5:
                    Level3Index = 99
                elif chroma > 8 and chroma <= 11 and value >= 5.5:
                    Level3Index = 100
                elif chroma > 11 and value >= 5.5:
                    Level3Index = 97
                elif chroma > 3 and value >= 4.5 and value < 6.5:
                    Level3Index = 106
                elif chroma > 3 and value >= 2.5 and value < 4.5:
                    Level3Index = 107
                elif chroma > 0.5 and value < 2.5:
                    Level3Index = 108
                else:
                    print("no match")

            elif hue >= 29 and hue < 32:
                if chroma > 0.5 and chroma <= 1.2 and value >= 8.5:
                    Level3Index = 92
                elif chroma > 0.5 and chroma <= 1.2 and value >= 6.5:
                    Level3Index = 93
                elif chroma > 0.5 and chroma <= 1.2 and value >= 4.5 and value < 6.5:
                    Level3Index = 112
                elif chroma > 0.5 and chroma <= 1.2 and value >= 2.5 and value < 4.5:
                    Level3Index = 113
                elif chroma > 1.2 and chroma <= 3 and value >= 7.5:
                    Level3Index = 121
                elif chroma > 1.2 and chroma <= 3 and value >= 6.5:
                    Level3Index = 122
                elif chroma > 1.2 and chroma <= 3 and value >= 4.5 and value < 6.5:
                    Level3Index = 109
                elif chroma > 1.2 and chroma <= 3 and value >= 2.5 and value < 4.5:
                    Level3Index = 110
                elif chroma > 0.5 and chroma <= 3 and value >= 1.5 and value < 2.5:
                    Level3Index = 111
                elif chroma > 0.5 and chroma <= 1 and value < 1.5:
                    Level3Index = 114
                elif chroma > 3 and chroma <= 5 and value >= 8:
                    Level3Index = 104
                elif chroma > 3 and chroma <= 5 and value >= 6.5:
                    Level3Index = 105
                elif chroma > 5 and chroma <= 8 and value >= 8:
                    Level3Index = 101
                elif chroma > 5 and chroma <= 8 and value >= 6.5:
                    Level3Index = 102
                elif chroma > 5 and chroma <= 8 and value >= 5.5:
                    Level3Index = 103
                elif chroma > 8 and chroma <= 11 and value >= 8:
                    Level3Index = 98
                elif chroma > 8 and chroma <= 11 and value >= 6.5:
                    Level3Index = 99
                elif chroma > 8 and chroma <= 11 and value >= 5.5:
                    Level3Index = 100
                elif chroma > 11 and value >= 5.5:
                    Level3Index = 97
                elif chroma > 3 and value >= 4.5 and value < 6.5:
                    Level3Index = 106
                elif chroma > 3 and value >= 2.5 and value < 4.5:
                    Level3Index = 107
                elif chroma > 0.5 and value < 2.5:
                    Level3Index = 108
                else:
                    print("no match")

            elif hue >= 32 and hue < 34:
                if chroma > 0.5 and chroma <= 1.2 and value >= 8.5:
                    Level3Index = 92
                elif chroma > 0.5 and chroma <= 1.2 and value >= 6.5:
                    Level3Index = 93
                elif chroma > 0.5 and chroma <= 1.2 and value >= 4.5 and value < 6.5:
                    Level3Index = 112
                elif chroma > 0.5 and chroma <= 1.2 and value >= 2.5 and value < 4.5:
                    Level3Index = 113
                elif chroma > 1.2 and chroma <= 3 and value >= 7.5:
                    Level3Index = 121
                elif chroma > 1.2 and chroma <= 3 and value >= 4.5:
                    Level3Index = 122
                elif chroma > 1.2 and chroma <= 3 and value >= 2.5:
                    Level3Index = 127
                elif chroma > 0.5 and chroma <= 3 and value >= 1.5 and value < 2.5:
                    Level3Index = 128
                elif chroma > 0.5 and chroma <= 1 and value < 1.5:
                    Level3Index = 114
                elif chroma > 3 and chroma <= 7 and value >= 7.5:
                    Level3Index = 119
                elif chroma > 3 and chroma <= 7 and value >= 4.5:
                    Level3Index = 120
                elif chroma > 3 and chroma <= 7 and value >= 2.5:
                    Level3Index = 125
                elif chroma > 7 and chroma <= 11 and value >= 7.5:
                    Level3Index = 116
                elif chroma > 7 and chroma <= 11 and value >= 4.5:
                    Level3Index = 117
                elif chroma > 7 and chroma <= 11 and value >= 3.5:
                    Level3Index = 118
                elif chroma > 7 and value >= 2.5 and value < 3.5:
                    Level3Index = 123
                elif chroma > 11 and value >= 3.5:
                    Level3Index = 115
                elif chroma > 7 and value < 2.5:
                    Level3Index = 124
                elif chroma > 1 and value < 2.5:
                    Level3Index = 126
                else:
                    print("no match")
                end

            elif hue >= 34 and hue < 38:
                if chroma > 0.5 and chroma <= 1.2 and value >= 8.5:
                    Level3Index = 153
                elif chroma > 0.5 and chroma <= 1.2 and value >= 6.5:
                    Level3Index = 154
                elif chroma > 0.5 and chroma <= 1.2 and value >= 4.5 and value < 6.5:
                    Level3Index = 155
                elif chroma > 0.5 and chroma <= 1.2 and value >= 2.5 and value < 4.5:
                    Level3Index = 156
                elif chroma > 1.2 and chroma <= 3 and value >= 7.5:
                    Level3Index = 121
                elif chroma > 1.2 and chroma <= 3 and value >= 4.5:
                    Level3Index = 122
                elif chroma > 1.2 and chroma <= 3 and value >= 2.5:
                    Level3Index = 127
                elif chroma > 0.5 and chroma <= 3 and value >= 1.5 and value < 2.5:
                    Level3Index = 128
                elif chroma > 0.5 and chroma <= 1 and value < 1.5:
                    Level3Index = 157
                elif chroma > 3 and chroma <= 7 and value >= 7.5:
                    Level3Index = 119
                elif chroma > 3 and chroma <= 7 and value >= 4.5:
                    Level3Index = 120
                elif chroma > 3 and chroma <= 7 and value >= 2.5:
                    Level3Index = 125
                elif chroma > 7 and chroma <= 11 and value >= 7.5:
                    Level3Index = 116
                elif chroma > 7 and chroma <= 11 and value >= 4.5:
                    Level3Index = 117
                elif chroma > 7 and chroma <= 11 and value >= 3.5:
                    Level3Index = 118
                elif chroma > 7 and value >= 2.5 and value < 3.5:
                    Level3Index = 123
                elif chroma > 11 and value >= 3.5:
                    Level3Index = 115
                elif chroma > 7 and value < 2.5:
                    Level3Index = 124
                elif chroma > 1 and value < 2.5:
                    Level3Index = 126
                else:
                    print("no match")
                end

            elif hue >= 38 and hue < 43:
                if chroma > 0.5 and chroma <= 1.2 and value >= 8.5:
                    Level3Index = 153
                elif chroma > 0.5 and chroma <= 1.2 and value >= 6.5:
                    Level3Index = 154
                elif chroma > 0.5 and chroma <= 1.2 and value >= 4.5 and value < 6.5:
                    Level3Index = 155
                elif chroma > 0.5 and chroma <= 1.2 and value >= 2.5 and value < 4.5:
                    Level3Index = 156
                elif chroma > 1.2 and chroma <= 2.5 and value >= 7.5:
                    Level3Index = 148
                elif chroma > 1.2 and chroma <= 2.5 and value >= 5.5:
                    Level3Index = 149
                elif chroma > 1.2 and chroma <= 2.5 and value >= 3.5:
                    Level3Index = 150
                elif (chroma > 1 and chroma <= 2.5 and value >= 2.5 and value < 3.5) or (chroma > 0.5 and chroma <= 2 and value >= 2 and value < 2.5):
                    Level3Index = 151
                elif chroma > 0.5 and chroma <= 1 and value < 2:
                    Level3Index = 157
                elif chroma > 1 and chroma <= 2 and value < 2:
                    Level3Index = 152
                elif chroma > 2.5 and chroma <= 7 and value >= 8.5:
                    Level3Index = 134
                elif chroma > 2.5 and chroma <= 7 and value >= 6.5:
                    Level3Index = 135
                elif chroma > 2.5 and chroma <= 7 and value >= 4.5:
                    Level3Index = 136
                elif chroma > 2.5 and chroma <= 7 and value >= 2.5:
                    Level3Index = 137
                elif chroma > 2 and chroma <= 7 and value < 2.5:
                    Level3Index = 138
                elif chroma > 7 and chroma <= 11 and value >= 6.5:
                    Level3Index = 130
                elif chroma > 7 and chroma <= 11 and value >= 4.5:
                    Level3Index = 131
                elif chroma > 7 and value >= 2.5 and value <= 4.5:
                    Level3Index = 132
                elif chroma > 7 and value < 2.5:
                    Level3Index = 133
                elif chroma > 11 and value > 4.5:
                    Level3Index = 129
                else:
                    print("no match")

            elif hue >= 43 and hue < 49:
                if chroma > 0.5 and chroma <= 1.2 and value >= 8.5:
                    Level3Index = 153
                elif chroma > 0.5 and chroma <= 1.2 and value >= 6.5:
                    Level3Index = 154
                elif chroma > 0.5 and chroma <= 1.2 and value >= 4.5 and value < 6.5:
                    Level3Index = 155
                elif chroma > 0.5 and chroma <= 1.2 and value >= 2.5 and value < 4.5:
                    Level3Index = 156
                elif chroma > 1.2 and chroma <= 2.5 and value >= 7.5:
                    Level3Index = 148
                elif chroma > 1.2 and chroma <= 2.5 and value >= 5.5:
                    Level3Index = 149
                elif chroma > 1.2 and chroma <= 2.5 and value >= 3.5:
                    Level3Index = 150
                elif (chroma > 1 and chroma <= 2.5 and value >= 2.5 and value < 3.5) or (chroma > 0.5 and chroma <= 2 and value >= 2 and value < 2.5):
                    Level3Index = 151
                elif chroma > 0.5 and chroma <= 1 and value < 2:
                    Level3Index = 157
                elif chroma > 1 and chroma <= 2 and value < 2:
                    Level3Index = 152
                elif chroma > 2.5 and chroma <= 7 and value >= 7.5:
                    Level3Index = 143
                elif chroma > 2.5 and chroma <= 7 and value >= 5.5:
                    Level3Index = 144
                elif chroma > 2.5 and chroma <= 7 and value >= 3.5:
                    Level3Index = 145
                elif chroma > 2 and chroma <= 7 and value >= 2:
                    Level3Index = 146
                elif chroma > 2 and chroma <= 7 and value < 2:
                    Level3Index = 147
                elif chroma > 7 and chroma <= 11 and value >= 5.5:
                    Level3Index = 140
                elif chroma > 7 and chroma <= 11 and value >= 3.5:
                    Level3Index = 141
                elif chroma > 7 and chroma <= 11 and value < 3.5:
                    Level3Index = 142
                elif chroma > 11:
                    Level3Index = 139
                else:
                    print("no match")

            elif hue >= 49 and hue < 60:
                if chroma > 0.5 and chroma <= 1.2 and value >= 8.5:
                    Level3Index = 153
                elif chroma > 0.5 and chroma <= 1.2 and value >= 6.5:
                    Level3Index = 154
                elif chroma > 0.5 and chroma <= 1.2 and value >= 4.5 and value < 6.5:
                    Level3Index = 155
                elif chroma > 0.5 and chroma <= 1.2 and value >= 2.5 and value < 4.5:
                    Level3Index = 156
                elif chroma > 1.2 and chroma <= 2.5 and value >= 7.5:
                    Level3Index = 148
                elif chroma > 1.2 and chroma <= 2.5 and value >= 5.5:
                    Level3Index = 149
                elif chroma > 1.2 and chroma <= 2.5 and value >= 3.5:
                    Level3Index = 150
                elif (chroma > 1.2 and chroma <= 2.5 and value >= 2.5 and value < 3.5) or (chroma > 0.5 and chroma <= 2 and value >= 2 and value < 2.5):
                    Level3Index = 151
                elif chroma > 0.5 and chroma <= 1 and value < 2:
                    Level3Index = 157
                elif chroma > 1 and chroma <= 2 and value < 2:
                    Level3Index = 152
                elif chroma > 2.5 and chroma <= 7 and value >= 7.5:
                    Level3Index = 162
                elif chroma > 2.5 and chroma <= 7 and value >= 5.5:
                    Level3Index = 163
                elif chroma > 2.5 and chroma <= 7 and value >= 3.5:
                    Level3Index = 164
                elif chroma > 2 and chroma <= 7 and value >= 2:
                    Level3Index = 165
                elif chroma > 2 and chroma <= 7 and value < 2:
                    Level3Index = 166
                elif chroma > 7 and chroma <= 11 and value >= 5.5:
                    Level3Index = 159
                elif chroma > 7 and chroma <= 11 and value >= 3.5:
                    Level3Index = 160
                elif chroma > 7 and chroma <= 11 and value < 3.5:
                    Level3Index = 161
                elif chroma > 11:
                    Level3Index = 158
                else:
                    print("no match")

            elif hue >= 60 and hue < 69:
                if chroma > 0.5 and chroma <= 1.5 and value >= 8.5:
                    Level3Index = 189
                elif chroma > 0.5 and chroma <= 1.5 and value >= 6.5:
                    Level3Index = 190
                elif chroma > 0.5 and chroma <= 1.5 and value >= 4.5 and value < 6.5:
                    Level3Index = 191
                elif chroma > 0.5 and chroma <= 1.5 and value >= 2.5 and value < 4.5:
                    Level3Index = 192
                elif chroma > 1.5 and chroma <= 3 and value >= 7.5:
                    Level3Index = 184
                elif chroma > 1.5 and chroma <= 3 and value >= 5.5:
                    Level3Index = 185
                elif chroma > 1.5 and chroma <= 3 and value >= 3:
                    Level3Index = 186
                elif (chroma > 1.5 and chroma <= 3 and value >= 2.5 and value < 3) or (chroma > 0.5 and chroma <= 2 and value >= 2 and value < 2.5):
                    Level3Index = 187
                elif chroma > 0.5 and chroma <= 1 and value < 2:
                    Level3Index = 193
                elif chroma > 1 and chroma <= 2 and value < 2:
                    Level3Index = 188
                elif chroma > 3 and chroma <= 7 and value >= 7.5:
                    Level3Index = 171
                elif chroma > 3 and chroma <= 7 and value >= 5.5:
                    Level3Index = 172
                elif chroma > 2 and chroma <= 7 and value >= 3.5:
                    Level3Index = 173
                elif chroma > 2 and chroma <= 7 and value >= 2:
                    Level3Index = 174
                elif chroma > 2 and chroma <= 7 and value < 2:
                    Level3Index = 175
                elif chroma > 7 and chroma <= 11 and value >= 5.5:
                    Level3Index = 168
                elif chroma > 7 and chroma <= 11 and value >= 3.5:
                    Level3Index = 169
                elif chroma > 7 and chroma <= 11 and value < 3.5:
                    Level3Index = 170
                elif chroma > 11:
                    Level3Index = 167
                else:
                    print("no match")

            elif hue >= 69 and hue < 75:
                if chroma > 0.5 and chroma <= 1.5 and value >= 8.5:
                    Level3Index = 189
                elif chroma > 0.5 and chroma <= 1.5 and value >= 6.5:
                    Level3Index = 190
                elif chroma > 0.5 and chroma <= 1.5 and value >= 4.5 and value < 6.5:
                    Level3Index = 191
                elif chroma > 0.5 and chroma <= 1.5 and value >= 2.5 and value < 4.5:
                    Level3Index = 192
                elif chroma > 1.5 and chroma <= 5 and value >= 7.5:
                    Level3Index = 184
                elif chroma > 1.5 and chroma <= 5 and value >= 5.5:
                    Level3Index = 185
                elif chroma > 1.5 and chroma <= 5 and value >= 3:
                    Level3Index = 186
                elif (chroma > 1.5 and chroma <= 3 and value >= 2.5 and value < 3) or (chroma > 0.5 and chroma <= 2 and value >= 2 and value < 2.5):
                    Level3Index = 187
                elif chroma > 0.5 and chroma <= 1 and value < 2:
                    Level3Index = 193
                elif chroma > 1 and chroma <= 2 and value < 2:
                    Level3Index = 188
                elif chroma > 5 and chroma <= 9 and value >= 7.5:
                    Level3Index = 180
                elif chroma > 5 and chroma <= 9 and value >= 5.5:
                    Level3Index = 181
                elif chroma > 5 and chroma <= 9 and value >= 3:
                    Level3Index = 182
                elif chroma > 9 and chroma <= 13 and value >= 5.5:
                    Level3Index = 177
                elif chroma > 9 and chroma <= 13 and value > 3:
                    Level3Index = 178
                elif chroma > 2 and chroma <= 7 and value <= 3:
                    Level3Index = 183
                elif chroma > 7 and chroma <= 11 and value <= 3:
                    Level3Index = 179
                elif (chroma > 13) or (chroma > 11 and value <= 3):
                    Level3Index = 176
                else:
                    print("no match")

            elif hue >= 75 and hue < 76:
                if chroma > 0.5 and chroma <= 1.5 and value >= 8.5:
                    Level3Index = 189
                elif chroma > 0.5 and chroma <= 1.5 and value >= 6.5:
                    Level3Index = 190
                elif chroma > 0.5 and chroma <= 1.5 and value >= 4.5 and value < 6.5:
                    Level3Index = 191
                elif chroma > 0.5 and chroma <= 1.5 and value >= 2.5 and value < 4.5:
                    Level3Index = 192
                elif chroma > 1.5 and chroma <= 3 and value >= 7.5:
                    Level3Index = 184
                elif chroma > 1.5 and chroma <= 3 and value >= 5.5:
                    Level3Index = 185
                elif chroma > 1.5 and chroma <= 3 and value >= 3:
                    Level3Index = 186
                elif (chroma > 1.5 and chroma <= 3 and value >= 2.5 and value < 3) or (chroma > 0.5 and chroma <= 2 and value >= 2 and value < 2.5):
                    Level3Index = 187
                elif chroma > 0.5 and chroma <= 1 and value < 2:
                    Level3Index = 193
                elif chroma > 1 and chroma <= 2 and value < 2:
                    Level3Index = 188
                elif chroma > 3 and chroma <= 5 and value >= 7.5:
                    Level3Index = 202
                elif chroma > 3 and chroma <= 5 and value >= 4.5:
                    Level3Index = 203
                elif chroma > 3 and chroma <= 5 and value >= 3:
                    Level3Index = 204
                elif chroma > 5 and chroma <= 7 and value >= 7.5:
                    Level3Index = 198
                elif chroma > 5 and chroma <= 7 and value >= 4.5:
                    Level3Index = 199
                elif chroma > 7 and chroma <= 9 and value >= 7.5:
                    Level3Index = 180
                elif chroma > 7 and chroma <= 9 and value >= 5.5:
                    Level3Index = 181
                elif chroma > 5 and chroma <= 9 and value >= 3:
                    Level3Index = 182
                elif chroma > 9 and chroma <= 13 and value >= 5.5:
                    Level3Index = 177
                elif chroma > 9 and chroma <= 13 and value > 3:
                    Level3Index = 178
                elif chroma > 2 and chroma <= 7 and value <= 3:
                    Level3Index = 183
                elif chroma > 7 and chroma <= 11 and value <= 3:
                    Level3Index = 179
                elif (chroma > 13) or (chroma > 11 and value <= 3):
                    Level3Index = 176
                else:
                    print("no match")

            elif hue >= 76 and hue < 77:
                if chroma > 0.5 and chroma <= 1.5 and value >= 8.5:
                    Level3Index = 189
                elif chroma > 0.5 and chroma <= 1.5 and value >= 6.5:
                    Level3Index = 190
                elif chroma > 0.5 and chroma <= 1.5 and value >= 4.5 and value < 6.5:
                    Level3Index = 191
                elif chroma > 0.5 and chroma <= 1.5 and value >= 2.5 and value < 4.5:
                    Level3Index = 192
                elif chroma > 1.5 and chroma <= 3 and value >= 7.5:
                    Level3Index = 184
                elif chroma > 1.5 and chroma <= 3 and value >= 5.5:
                    Level3Index = 185
                elif chroma > 1.5 and chroma <= 3 and value >= 3:
                    Level3Index = 186
                elif (chroma > 1.5 and chroma <= 3 and value >= 2.5 and value < 3) or (chroma > 0.5 and chroma <= 2 and value >= 2 and value < 2.5):
                    Level3Index = 187
                elif chroma > 0.5 and chroma <= 1 and value < 2:
                    Level3Index = 193
                elif chroma > 1 and chroma <= 2 and value < 2:
                    Level3Index = 188
                elif chroma > 3 and chroma <= 5 and value >= 7.5:
                    Level3Index = 202
                elif chroma > 3 and chroma <= 5 and value >= 4.5:
                    Level3Index = 203
                elif chroma > 2 and chroma <= 5 and value >= 2:
                    Level3Index = 204
                elif chroma > 2 and chroma <= 5 and value <= 2:
                    Level3Index = 201
                elif chroma > 5 and chroma <= 7 and value >= 7.5:
                    Level3Index = 198
                elif chroma > 5 and chroma <= 7 and value >= 4.5:
                    Level3Index = 199
                elif chroma > 7 and chroma <= 9 and value >= 7.5:
                    Level3Index = 180
                elif chroma > 7 and chroma <= 9 and value >= 5.5:
                    Level3Index = 181
                elif chroma > 5 and chroma <= 9 and value >= 3:
                    Level3Index = 182
                elif chroma > 9 and chroma <= 13 and value >= 5.5:
                    Level3Index = 177
                elif chroma > 9 and chroma <= 13 and value > 3:
                    Level3Index = 178
                elif chroma > 2 and chroma <= 7 and value <= 3:
                    Level3Index = 183
                elif chroma > 7 and chroma <= 11 and value <= 3:
                    Level3Index = 179
                elif (chroma > 13) or (chroma > 11 and value <= 3):
                    Level3Index = 176
                else:
                    print("no match")

            elif hue >= 77 and hue < 79:
                if chroma > 0.5 and chroma <= 1.5 and value >= 8.5:
                    Level3Index = 189
                elif chroma > 0.5 and chroma <= 1.5 and value >= 6.5:
                    Level3Index = 190
                elif chroma > 0.5 and chroma <= 1.5 and value >= 4.5 and value < 6.5:
                    Level3Index = 191
                elif chroma > 0.5 and chroma <= 1.5 and value >= 2.5 and value < 4.5:
                    Level3Index = 192
                elif chroma > 1.5 and chroma <= 3 and value >= 7.5:
                    Level3Index = 184
                elif chroma > 1.5 and chroma <= 3 and value >= 5.5:
                    Level3Index = 185
                elif chroma > 1.5 and chroma <= 3 and value >= 3:
                    Level3Index = 186
                elif (chroma > 1.5 and chroma <= 3 and value >= 2.5 and value < 3) or (chroma > 0.5 and chroma <= 2 and value >= 2 and value < 2.5):
                    Level3Index = 187
                elif chroma > 0.5 and chroma <= 1 and value < 2:
                    Level3Index = 193
                elif chroma > 1 and chroma <= 2 and value < 2:
                    Level3Index = 188
                elif chroma > 3 and chroma <= 5 and value >= 7.5:
                    Level3Index = 202
                elif chroma > 3 and chroma <= 5 and value >= 4.5:
                    Level3Index = 203
                elif chroma > 2 and chroma <= 5 and value >= 2:
                    Level3Index = 204
                elif chroma > 2 and chroma <= 7 and value <= 2:
                    Level3Index = 201
                elif chroma > 5 and chroma <= 9 and value >= 7.5:
                    Level3Index = 198
                elif chroma > 5 and chroma <= 9 and value >= 4.5:
                    Level3Index = 199
                elif (chroma > 5 and chroma <= 9 and value >= 3 and value < 4.5) or (chroma > 5 and chroma <= 7 and value >= 2 and value < 3):
                    Level3Index = 200
                elif (chroma > 9 and chroma <= 13 and value >= 5.5) or (chroma > 9 and chroma <= 11 and value >= 4.5 and value < 5.5):
                    Level3Index = 195
                elif chroma > 9 and chroma <= 13 and value > 3:
                    Level3Index = 196
                elif chroma > 7 and chroma <= 11 and value <= 3:
                    Level3Index = 197
                elif (chroma > 13) or (chroma > 11 and value <= 3):
                    Level3Index = 194
                else:
                    print("no match")

            elif hue >= 79 and hue < 83:
                if chroma > 0.5 and chroma <= 1.5 and value >= 8.5:
                    Level3Index = 231
                elif chroma > 0.5 and chroma <= 1.5 and value >= 6.5:
                    Level3Index = 232
                elif chroma > 0.5 and chroma <= 1.5 and value >= 4.5 and value < 6.5:
                    Level3Index = 233
                elif chroma > 0.5 and chroma <= 1.5 and value >= 2.5 and value < 4.5:
                    Level3Index = 234
                elif chroma > 1.5 and chroma <= 3 and value >= 7.5:
                    Level3Index = 226
                elif chroma > 1.5 and chroma <= 3 and value >= 5.5:
                    Level3Index = 227
                elif chroma > 1.5 and chroma <= 3 and value >= 3.5:
                    Level3Index = 228
                elif (chroma > 1.5 and chroma <= 3 and value >= 2.5 and value < 3.5) or (chroma > 0.5 and chroma <= 2 and value >= 2 and value < 2.5):
                    Level3Index = 229
                elif chroma > 0.5 and chroma <= 1 and value < 2:
                    Level3Index = 235
                elif chroma > 1 and chroma <= 2 and value < 2:
                    Level3Index = 230
                elif chroma > 3 and chroma <= 5 and value >= 7.5:
                    Level3Index = 213
                elif chroma > 3 and chroma <= 5 and value >= 4.5:
                    Level3Index = 214
                elif chroma > 3 and chroma <= 5 and value >= 2.5:
                    Level3Index = 215
                elif chroma > 2 and chroma <= 7 and value <= 2.5:
                    Level3Index = 212
                elif chroma > 5 and chroma <= 9 and value >= 7.5:
                    Level3Index = 209
                elif chroma > 5 and chroma <= 9 and value >= 4.5:
                    Level3Index = 210
                elif chroma > 5 and chroma <= 9 and value >= 2.5:
                    Level3Index = 211
                elif chroma > 9 and chroma <= 13 and value >= 4.5:
                    Level3Index = 206
                elif chroma > 9 and chroma <= 13 and value >= 2.5:
                    Level3Index = 207
                elif chroma > 7 and chroma <= 13 and value <= 2.5:
                    Level3Index = 208
                elif chroma > 13:
                    Level3Index = 205
                else:
                    print("no match")

            elif hue >= 83 and hue < 89:
                if chroma > 0.5 and chroma <= 1.5 and value >= 8.5:
                    Level3Index = 231
                elif chroma > 0.5 and chroma <= 1.5 and value >= 6.5:
                    Level3Index = 232
                elif chroma > 0.5 and chroma <= 1.5 and value >= 4.5 and value < 6.5:
                    Level3Index = 233
                elif chroma > 0.5 and chroma <= 1.5 and value >= 2.5 and value < 4.5:
                    Level3Index = 234
                elif (chroma > 1.5 and chroma <= 3 and value >= 2.5 and value < 3.5) or (chroma > 0.5 and chroma <= 2 and value >= 2 and value < 2.5):
                    Level3Index = 229
                elif chroma > 1.5 and chroma <= 5 and value >= 7.5:
                    Level3Index = 226
                elif chroma > 1.5 and chroma <= 5 and value >= 5.5:
                    Level3Index = 227
                elif chroma > 1.5 and chroma <= 5 and value >= 3.5:
                    Level3Index = 228
                elif chroma > 5 and chroma <= 9 and value >= 7.5:
                    Level3Index = 221
                elif chroma > 5 and chroma <= 9 and value >= 5.5:
                    Level3Index = 222
                elif chroma > 5 and chroma <= 9 and value >= 3.5:
                    Level3Index = 223
                elif chroma > 9 and chroma <= 13 and value >= 5.5:
                    Level3Index = 217
                elif chroma > 9 and chroma <= 13 and value >= 3.5:
                    Level3Index = 218
                elif chroma > 0.5 and chroma <= 1 and value < 2:
                    Level3Index = 235
                elif chroma > 1 and chroma <= 2 and value < 2:
                    Level3Index = 230
                elif chroma > 2 and chroma <= 7 and value >= 2 and value < 3.5:
                    Level3Index = 224
                elif chroma > 2 and chroma <= 7 and value < 2:
                    Level3Index = 225
                elif chroma > 7 and chroma <= 13 and value >= 2 and value < 3.5:
                    Level3Index = 219
                elif chroma > 7 and chroma <= 13 and value < 2:
                    Level3Index = 220
                elif chroma > 13:
                    Level3Index = 216
                else:
                    print("no match")

            elif hue >= 89 and hue < 93:
                if chroma > 0.5 and chroma <= 1.5 and value >= 8.5:
                    Level3Index = 231
                elif chroma > 0.5 and chroma <= 1.5 and value >= 6.5:
                    Level3Index = 232
                elif chroma > 0.5 and chroma <= 1.5 and value >= 4.5 and value < 6.5:
                    Level3Index = 233
                elif chroma > 0.5 and chroma <= 1.5 and value >= 2.5 and value < 4.5:
                    Level3Index = 234
                elif (chroma > 1.5 and chroma <= 3 and value >= 2.5 and value < 3.5) or (chroma > 0.5 and chroma <= 2 and value >= 2 and value < 2.5):
                    Level3Index = 229
                elif chroma > 0.5 and chroma <= 1 and value < 2:
                    Level3Index = 235
                elif chroma > 1 and chroma <= 2 and value < 2:
                    Level3Index = 230
                elif chroma > 1.5 and chroma <= 5 and value >= 7.5:
                    Level3Index = 252
                elif chroma > 1.5 and chroma <= 5 and value >= 6.5:
                    Level3Index = 253
                elif chroma > 1.5 and chroma <= 3 and value >= 5.5:
                    Level3Index = 227
                elif chroma > 1.5 and chroma <= 3 and value >= 3.5:
                    Level3Index = 228
                elif chroma > 3 and chroma <= 5 and value >= 5.5 and value < 6.5:
                    Level3Index = 244
                elif chroma > 3 and chroma <= 5 and value >= 3.5 and value < 5.5:
                    Level3Index = 245
                elif chroma > 5 and chroma <= 9 and value >= 7.5:
                    Level3Index = 249
                elif chroma > 5 and chroma <= 9 and value >= 6.5:
                    Level3Index = 250
                elif chroma > 5 and chroma <= 9 and value >= 5.5:
                    Level3Index = 240
                elif chroma > 5 and chroma <= 9 and value >= 3.5:
                    Level3Index = 241
                elif chroma > 9 and value >= 7.5:
                    Level3Index = 246
                elif chroma > 9 and value >= 6.5 and value < 7.5:
                    Level3Index = 247
                elif chroma > 9 and chroma <= 15 and value >= 5.5 and value < 6.5:
                    Level3Index = 248
                elif chroma > 9 and chroma <= 13 and value >= 3.5 and value < 5.5:
                    Level3Index = 237
                elif chroma > 13 and value < 6.5:
                    Level3Index = 236
                elif chroma > 2 and chroma <= 7 and value >= 2 and value < 3.5:
                    Level3Index = 242
                elif chroma > 2 and chroma <= 7 and value < 2:
                    Level3Index = 243
                elif chroma > 7 and chroma <= 13 and value >= 2 and value < 3.5:
                    Level3Index = 238
                elif chroma > 7 and chroma <= 13 and value < 2:
                    Level3Index = 239
                else:
                    print("no match")

            elif hue >= 93 and hue < 99:
                if chroma > 0.5 and chroma <= 1.5 and value >= 8.5:
                    Level3Index = 231
                elif chroma > 0.5 and chroma <= 1.5 and value >= 6.5:
                    Level3Index = 232
                elif chroma > 0.5 and chroma <= 1.5 and value >= 4.5 and value < 6.5:
                    Level3Index = 233
                elif chroma > 0.5 and chroma <= 1.5 and value >= 2.5 and value < 4.5:
                    Level3Index = 234
                elif (chroma > 1.5 and chroma <= 3 and value >= 2.5 and value < 3.5) or (chroma > 0.5 and chroma <= 2 and value >= 2 and value < 2.5):
                    Level3Index = 229
                elif chroma > 0.5 and chroma <= 1 and value < 2:
                    Level3Index = 235
                elif chroma > 1 and chroma <= 2 and value < 2:
                    Level3Index = 230
                elif chroma > 1.5 and chroma <= 5 and value >= 7.5:
                    Level3Index = 252
                elif chroma > 1.5 and chroma <= 5 and value >= 6.5:
                    Level3Index = 253
                elif chroma > 1.5 and chroma <= 3 and value >= 5.5:
                    Level3Index = 227
                elif chroma > 1.5 and chroma <= 3 and value >= 3.5:
                    Level3Index = 228
                elif chroma > 3 and chroma <= 5 and value >= 5.5 and value < 6.5:
                    Level3Index = 261
                elif chroma > 3 and chroma <= 7 and value >= 3.5 and value < 5.5:
                    Level3Index = 262
                elif chroma > 5 and chroma <= 9 and value >= 7.5:
                    Level3Index = 249
                elif chroma > 5 and chroma <= 9 and value >= 6.5:
                    Level3Index = 250
                elif chroma > 5 and chroma <= 9 and value >= 5.5:
                    Level3Index = 251
                elif chroma > 7 and chroma <= 11 and value >= 3.5 and value < 5.5:
                    Level3Index = 258
                elif chroma > 9 and value >= 7.5:
                    Level3Index = 246
                elif chroma > 9 and value >= 6.5 and value < 7.5:
                    Level3Index = 247
                elif chroma > 9 and chroma <= 15 and value >= 5.5 and value < 6.5:
                    Level3Index = 248
                elif chroma > 11 and chroma <= 13 and value >= 3.5 and value < 5.5:
                    Level3Index = 255
                elif chroma > 2 and chroma <= 9 and value >= 2 and value < 3.5:
                    Level3Index = 259
                elif chroma > 2 and chroma <= 7 and value < 2:
                    Level3Index = 260
                elif chroma > 9 and chroma <= 13 and value >= 2 and value < 3.5:
                    Level3Index = 256
                elif chroma > 7 and chroma <= 11 and value < 2:
                    Level3Index = 257
                elif chroma > 11 and value < 6.5:
                    Level3Index = 254
                else:
                    print("no match")

            elif hue >= 99 or hue < 1:
                if chroma > 0.5 and chroma <= 1.5 and value >= 8.5:
                    Level3Index = 9
                elif chroma > 0.5 and chroma <= 1.5 and value >= 6.5:
                    Level3Index = 10
                elif chroma > 0.5 and chroma <= 1.5 and value >= 4.5:
                    Level3Index = 233
                elif chroma > 0.5 and chroma <= 1.5 and value >= 2.5:
                    Level3Index = 234
                elif (chroma > 0.5 and chroma <= 2 and value >= 2 and value < 2.5) or (chroma > 1.5 and chroma <= 3 and value >= 2.5 and value < 3.5):
                    Level3Index = 229
                elif chroma > 0.5 and chroma <= 1 and value < 2:
                    Level3Index = 235
                elif chroma > 1 and chroma <= 2 and value < 2:
                    Level3Index = 230
                elif chroma > 1.5 and chroma <= 3 and value >= 8:
                    Level3Index = 7
                elif chroma > 1.5 and chroma <= 3 and value >= 6.5:
                    Level3Index = 8
                elif chroma > 1.5 and chroma <= 3 and value >= 5.5:
                    Level3Index = 227
                elif chroma > 1.5 and chroma <= 3 and value >= 3.5:
                    Level3Index = 228
                elif chroma > 3 and chroma <= 7 and value >= 8:
                    Level3Index = 4
                elif chroma > 3 and chroma <= 7 and value >= 6.5:
                    Level3Index = 5
                elif chroma > 7 and chroma <= 11 and value >= 6.5:
                    Level3Index = 2
                elif chroma > 11 and value >= 6.5:
                    Level3Index = 1
                elif chroma > 3 and chroma <= 5 and value >= 5.5 and value < 6.5:
                    Level3Index = 261
                elif chroma > 5 and chroma <= 7 and value >= 5.5 and value < 6.5:
                    Level3Index = 6
                elif chroma > 7 and chroma <= 15 and value >= 5.5 and value < 6.5:
                    Level3Index = 3
                elif chroma > 3 and chroma <= 7 and value >= 3.5 and value < 5.5:
                    Level3Index = 262
                elif chroma > 7 and chroma <= 11 and value >= 3.5 and value < 5.5:
                    Level3Index = 258
                elif chroma > 11 and chroma <= 13 and value >= 3.5 and value < 5.5:
                    Level3Index = 255
                elif chroma >= 2 and chroma <= 9 and value >= 2 and value < 3.5:
                    Level3Index = 259
                elif chroma > 9 and chroma <= 13 and value >= 2 and value < 3.5:
                    Level3Index = 256
                elif chroma > 2 and chroma <= 7 and value < 2:
                    Level3Index = 260
                elif chroma > 7 and chroma <= 11 and value < 2:
                    Level3Index = 257
                elif chroma > 11 and value < 6.5:
                    Level3Index = 254
                else:
                    print("no match")

        return Level3Index

