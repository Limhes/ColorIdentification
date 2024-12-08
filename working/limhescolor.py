import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
import colour
import cv2 as cv


def rgb2xyz(rgb, illuminant="D65"):
    # 2 degree observer:
    if illuminant == "D50":
        ill = [96.4212, 100.0, 82.5188]
    else: # D65 and default
        ill = [95.047, 100.0, 108.883]

    rgb = np.array(rgb, dtype=float) / 255.0
    rgb = np.where(rgb > 0.04045, ((rgb+0.055)/1.055)**2.4, rgb/12.92) * 100
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]], dtype=float)
    xyz = np.matmul(M, rgb)
    xyz = np.divide(xyz, ill)
    xyz = np.where(xyz > 0.008856, xyz**0.3333, (7.787*xyz)+(16/116))
    return [round(x, 4) for x in xyz]

def xyz2rgb(xyz, illuminant="D65"):
    # 2 degree observer:
    if illuminant == "D50":
        ill = [96.4212, 100.0, 82.5188]
    else: # D65 and default
        ill = [95.047, 100.0, 108.883]

    delta = 6.0/29.0
    xyz = np.array(xyz, dtype=float)
    xyz = np.where(xyz > delta, xyz**3, 3*(delta**2)*(xyz-4.0/29.0))
    xyz = np.multiply(xyz, ill)
    M = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]], dtype=float)
    rgb = np.matmul(M, xyz) / 100.0
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb = np.where(rgb > 0.0031308, (1.055*(rgb**(1.0/2.4))-0.055), rgb*12.92) * 255.0
    rgb = np.clip(rgb, 0.0, 255.0)
    return [round(x, 4) for x in rgb]

def xyz2lab(XYZ):
    return [round(x, 4) for x in [(116 * XYZ[1] ) - 16,
                                500 * ( XYZ[0] - XYZ[1]),
                                200 * ( XYZ[1] - XYZ[2])]]

def lab2xyz(LAB):
    return [round(x, 4) for x in [(LAB[0]+16.0)/116.0 + LAB[1]/500,
                                    (LAB[0]+16.0)/116.0,
                                    (LAB[0]+16.0)/116.0 - LAB[2]/200]]

def xyz2xyy(XYZ):
    return [round(x, 4) for x in [XYZ[0]/sum(XYZ),
                                XYZ[1]/sum(XYZ),
                                XYZ[1]]]
