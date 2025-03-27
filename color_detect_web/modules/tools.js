
function illuminant(ill) {
    // 2 degree observer:
    if (ill == "D50")
        return [96.4212, 100.0, 82.5188];
    else if (ill == "D65")
        return [95.047, 100.0, 108.883];
    else
        return [0, 0, 0];
}

function RGBtoLCH(rgb, _ill="D65") {
    const M = [[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]];
    const ill = illuminant(_ill);
    const xyz = [0, 0, 0];
    const lab = [0, 0, 0];
    const lch = [0, 0, 0];

    for (let x = 0; x < 3; x++) {
        rgb[x] /= 255.0;
        if (rgb[x] > 0.04045)
            rgb[x] = Math.pow((rgb[x]+0.055)/1.055, 2.4);
        else
            rgb[x] = rgb[x]/12.92;
        rgb[x] *= 100.0;
    }
    for (let x = 0; x < 3; x++) {
        for (let y = 0; y < 3; y++) {
            xyz[x] += rgb[y]*M[x][y];
        }
    }
    for (let x = 0; x < 3; x++) {
        xyz[x] /= ill[x];
        if (xyz[x] > 0.008856)
            xyz[x] = Math.pow(xyz[x], 0.3333);
        else
            xyz[x] = (7.787*xyz[x])+(16.0/116.0);
    }

    lab[0] = (116.0 * xyz[1]) - 16.0;
    lab[1] = 500.0 * (xyz[0] - xyz[1])
    lab[2] = 200.0 * (xyz[1] - xyz[2])

    lch[0] = lab[0] / 10.0;
    lch[1] = Math.sqrt(lab[1]*lab[1] + lab[2]*lab[2]) / 5.0;
    lch[2] = Math.atan2(lab[2], lab[1]) / Math.PI * 180.0;
    if (lch[2] < 0) lch[2] += 360.0;

    return lch;
}

function LCHtoRGB(lch, _ill="D65") {

    const M = [[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]];
    const ill = illuminant(_ill);
    const xyz = [0, 0, 0];
    const rgb = [0, 0, 0];
    const lab = [0, 0, 0];

    lab[0] = 10.0 * lch[0];
    lab[1] = 5.0 * lch[1] * Math.cos((lch[2]-180.0)/360.0*Math.PI);
    lab[2] = 5.0 * lch[1] * Math.sin((lch[2]-180.0)/360.0*Math.PI);

    xyz[0] = (lab[0] + 16.0) / 116.0 + lab[1] / 500.0
    xyz[1] = (lab[0] + 16.0) / 116.0
    xyz[2] = (lab[0] + 16.0) / 116.0 - lab[2] / 200.0

    for (let x = 0; x < 3; x++) {
        if (xyz[x] > 0.2069)
            xyz[x] = Math.pow(xyz[x], 3.0);
        else
            xyz[x] = 0.1284 * (xyz[x] - 4.0/29.0);

        xyz[x] *= ill[x];
    }
    for (let x = 0; x < 3; x++) {
        for (let y = 0; y < 3; y++) {
            rgb[x] += xyz[y]*M[x][y];
        }
    }
    for (let x = 0; x < 3; x++) {
        rgb[x] /= 100.0;

        if (rgb[x] > 0.0031308)
            rgb[x] = 1.055 * Math.pow(rgb[x], 1.0/2.4) - 0.055;
        else
            rgb[x] = rgb[x]*12.92;

        rgb[x] *= 255.0;
        rgb[x] = Math.max(0.0, Math.min(rgb[x], 255.0)); // clamp to 0,255
    }

    return rgb;
}

/* this function returns the [mean, stdev] of the input array */
function stdev(arr) {
    let mean = arr.reduce((acc, curr) => { return acc + curr; }, 0) / arr.length;
    arr = arr.map((k) => { return (k - mean) ** 2; });
    let sum = arr.reduce((acc, curr) => acc + curr, 0);
    return [mean, Math.sqrt(sum / arr.length)]
}

export {RGBtoLCH, LCHtoRGB, stdev};
