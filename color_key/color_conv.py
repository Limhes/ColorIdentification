import colour

def rgb2xyz(RGB, illuminant="D65"):
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
           RGB [0] * 0.0193 + RGB [1] * 0.1192 + RGB [2] * 0.9505]
    XYZ = [round(x, 4)/ill for x,ill in zip(XYZ,ill)]
    XYZ = [x**0.3333 if x > 0.008856 else (7.787*x)+(16/116) for x in XYZ]

    return [round(x, 4) for x in XYZ]

def xyz2lab(XYZ):
    return [round(x, 4) for x in [(116 * XYZ[1] ) - 16,
                                   500 * ( XYZ[0] - XYZ[1]),
                                   200 * ( XYZ[1] - XYZ[2])]]

def xyz2xyy(XYZ):
    return [round(x, 4) for x in [XYZ[0]/sum(XYZ),
                                  XYZ[1]/sum(XYZ),
                                  XYZ[1]]]

def xyy2munsell(xyY):
    return colour.xyY_to_munsell_colour(xyY)
