export default class ColorCharts extends HTMLElement {

    canvas_ids = ["canvas_CH", "canvas_LC"];
    lch_display = [[1, 2], [1, 0]]; // 2,1 = H,C; 1,0 = C,L
    count_picked_colors = 0;
    data = [
        [{
            r: [],
            theta: [],
            text: [],
            mode: 'markers+text',
            type: 'scatterpolar'
        }], [{
            x: [],
            y: [],
            text: [],
            mode: 'markers+text',
            type: 'scatter'
        }]];
    layout = [
        {
            autosize: true,
            font: { size: 10, family: "sans-serif" },
            margin: { l: 30, r: 30, b: 30, t: 30, pad: 4 },
            annotations: []
        },{
            autosize: true,
            font: { size: 10, family: "sans-serif" },
            margin: { l: 30, r: 10, b: 30, t: 10, pad: 4 },
            annotations: []
        }
    ];
    polar_annotations = [
        { text: 'red', angle: 28 },
        { text: 'orange', angle: 47 },
        { text: 'yellow', angle: 84 },
        { text: 'yellow-green', angle: 112 },
        { text: 'green', angle: 141 },
        { text: 'blue-green', angle: 194 },
        { text: 'greenish blue', angle: 231 },
        { text: 'blue', angle: 276 },
        { text: 'violet', angle: 319 },
        { text: 'purple', angle: 344 },
    ];

    constructor() {
        super();
    }

    connectedCallback() {
        const template = document.getElementById("color_charts_template");
        const clone = template.content.cloneNode(true);
        this.appendChild(clone);

        this.polar_annotations.forEach((ann) => {
            this.layout[0].annotations.push({
                text: ann.text,
                ax: 100*Math.cos((360 - ann.angle) / 180.0 * Math.PI),
                ay: 100*Math.sin((360 - ann.angle) / 180.0 * Math.PI)
            });
        });

        for (let c = 0; c < this.canvas_ids.length; ++c) {
            Plotly.newPlot(this.canvas_ids[c], this.data[c], this.layout[c]);
        }
    }

    redrawCharts() {
        for (let c = 0; c < this.canvas_ids.length; ++c) {
            Plotly.redraw(this.canvas_ids[c]);
        }
    }

    addColor(lch, lch_stdev, rgb) {
        let id = this.count_picked_colors++;
        let name = "Stamp " + String(id);

        this.data[0][0].r.push(lch[this.lch_display[0][0]]);
        this.data[0][0].theta.push(lch[this.lch_display[0][1]]);
        this.data[0][0].text.push(name);

        this.data[1][0].x.push(lch[this.lch_display[1][0]]);
        this.data[1][0].y.push(lch[this.lch_display[1][1]]);
        this.data[1][0].text.push(name);

        this.redrawCharts();
        return [id, name];
    }

    showColor(id, lch, lch_stdev, rgb, name) {
        this.data[0][0].r[id] = lch[this.lch_display[0][0]];
        this.data[0][0].theta[id] = lch[this.lch_display[0][1]];
        this.data[0][0].text[id] = name;

        this.data[1][0].x[id] = lch[this.lch_display[1][0]];
        this.data[1][0].y[id] = lch[this.lch_display[1][1]];
        this.data[1][0].text[id] = name;

        this.redrawCharts();
    }

    hideColor(id) {
        this.data[0][0].r[id] = null;
        this.data[0][0].theta[id] = null;
        this.data[0][0].text[id] = "";

        this.data[1][0].x[id] = null;
        this.data[1][0].y[id] = null;
        this.data[1][0].text[id] = "";

        this.redrawCharts();
    }
};
