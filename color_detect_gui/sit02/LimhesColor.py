import numpy as np


def illuminant(ill="D65"):
    # 2 degree observer:
    return np.array([96.4212, 100.0, 82.5188]) if illuminant == "D50" else np.array([95.047, 100.0, 108.883])

def rgb2xyz(rgb, ill="D65"):
    rgb = np.array(rgb, dtype=float).transpose() / 255.0
    rgb = np.where(rgb > 0.04045, ((rgb+0.055)/1.055)**2.4, rgb/12.92) * 100
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]], dtype=float)
    xyz = np.matmul(M, rgb).transpose()
    xyz = np.divide(xyz, illuminant(ill))
    xyz = np.where(xyz > 0.008856, xyz**0.3333, (7.787*xyz)+(16/116))
    return xyz

def xyz2rgb(xyz, ill="D65"):
    delta = 6.0/29.0
    xyz = np.array(xyz, dtype=float)
    xyz = np.where(xyz > delta, xyz**3, 3*(delta**2)*(xyz-4.0/29.0))
    xyz = np.multiply(xyz, illuminant(ill)).transpose()
    M = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]], dtype=float)
    rgb = np.matmul(M, xyz) / 100.0
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb = np.where(rgb > 0.0031308, (1.055*(rgb**(1.0/2.4))-0.055), rgb*12.92) * 255.0
    rgb = np.clip(rgb, 0.0, 255.0)
    return rgb.transpose()

def xyz2lab(xyz):
    lab = np.ones_like(xyz)
    lab[:,0] = ( 116.0 * xyz[:,1] ) - 16.0
    lab[:,1] = 500.0 * ( xyz[:,0] - xyz[:,1] )
    lab[:,2] = 200.0 * ( xyz[:,1] - xyz[:,2] )
    return lab

def lab2xyz(lab):
    xyz = np.ones_like(lab)
    xyz[:,0] = ( lab[:,0] + 16.0 ) / 116.0 + lab[:,1] / 500.0
    xyz[:,1] = ( lab[:,0] + 16.0 ) / 116.0
    xyz[:,2] = ( lab[:,0] + 16.0 ) / 116.0 - lab[:,2] / 200.0
    return xyz

def rgb2lab(rgb, illuminant="D65"):
    return xyz2lab(rgb2xyz(rgb, illuminant))

def lab2rgb(lab, illuminant="D65"):
    return xyz2rgb(lab2xyz(lab), illuminant)
