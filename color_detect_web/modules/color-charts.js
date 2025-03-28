export default class ColorCharts extends HTMLElement {

    canvas_ids = ["canvas_CH", "canvas_LC"];
    lch_display = [[1, 2], [1, 0]]; // 2,1 = H,C; 1,0 = C,L
    count_picked_colors = 0;
    data = [
        [{
            r: [],
            theta: [],
            textposition: 'top',
            text: [],
            mode: 'markers+text',
            type: 'scatterpolar'
        }], [{
            x: [],
            y: [],
            textposition: 'top',
            text: [],
            mode: 'markers+text',
            type: 'scatter'
        }]];
    layout = [
        {
            title: {
                text: 'Color (hue)',
                font: { size: 16 }
            },
            xaxis: {
                title: {
                  text: 'x Axis',
                  font: { size: 12 }
                },
            },
            yaxis: {
                title: {
                  text: 'x Axis',
                  font: { size: 12 }
                },
            },
            autosize: true,
            font: { size: 10, family: "sans-serif" },
            margin: { l: 30, r: 30, b: 30, t: 50, pad: 4 },
            annotations: []
        },{
            title: {
                text: 'Lightness (value) versus saturation (chroma)',
                font: { size: 16 },
                subtitle: {
                    text: 'Note that light purple/red/orange is called pink and dark yellow/orange/red is called brown.',
                    font: { size: 10 }
                }
            },
            xaxis: {
                title: {
                  text: 'Saturation (chroma)',
                  font: { size: 12 }
                },
            },
            yaxis: {
                title: {
                  text: 'Lightness (value)',
                  font: { size: 12 }
                },
            },
            autosize: true,
            font: { size: 10, family: "sans-serif" },
            margin: { l: 30, r: 10, b: 30, t: 50, pad: 4 },
            annotations: []
        }
    ];
    hue_annotations = [
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
    chromavalue_annotations = [
        { text: 'very light', x: 5, y: 8.5 },
        { text: 'light', x: 5, y: 6.5 },
        { text: 'moderate', x: 5, y: 4.5 },
        { text: 'dark', x: 5, y: 2.5 },
        { text: 'very dark', x: 5, y: 1 },
        { text: 'brilliant', x: 9, y: 7.5 },
        { text: 'deep', x: 9, y: 2 },
        { text: 'strong', x: 9, y: 4.5 },
        { text: 'vivid', x: 12, y: 5.5 },
        { text: 'grayish', x: 2, y: 4.5 },
        { text: 'pale', x: 2, y: 6.5 },
        { text: 'black', x: 0, y: 1 },
        { text: 'dark grey', x: 0, y: 3.5 },
        { text: 'medium grey', x: 0, y: 5.5 },
        { text: 'light grey', x: 0, y: 7.5 },
        { text: 'white', x: 0, y: 9 },
    ]

    constructor() {
        super();
    }

    connectedCallback() {
        const template = document.getElementById("color_charts_template");
        const clone = template.content.cloneNode(true);
        this.appendChild(clone);

        this.hue_annotations.forEach((ann) => {
            this.layout[0].annotations.push({
                text: ann.text,
                ax: 100*Math.cos((360 - ann.angle) / 180.0 * Math.PI),
                ay: 100*Math.sin((360 - ann.angle) / 180.0 * Math.PI),
                arrowwidth: 1,
                arrowcolor: '#888',
                font: {
                    color: '#888',
                    style: 'italic',
                },
            });
        });

        this.chromavalue_annotations.forEach((ann) => {
            this.layout[1].annotations.push({
                showarrow: false,
                text: ann.text,
                x: ann.x,
                y: ann.y,
                font: {
                    color: '#888',
                    style: 'italic',
                },
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

        //this.data[0][0].r.push(lch[this.lch_display[0][0]]);
        this.data[0][0].r.push(1);
        this.data[0][0].theta.push(lch[this.lch_display[0][1]]);
        this.data[0][0].text.push(name);

        this.data[1][0].x.push(lch[this.lch_display[1][0]]);
        this.data[1][0].y.push(lch[this.lch_display[1][1]]);
        this.data[1][0].text.push(name);

        this.redrawCharts();
        return [id, name];
    }

    showColor(id, lch, lch_stdev, rgb, name) {
        this.data[0][0].r[id] = 1; //lch[this.lch_display[0][0]];
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
