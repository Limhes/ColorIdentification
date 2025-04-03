import {LCHtoRGB, RGBtoLCH} from "./tools.js";

export default class ColorOrganizer extends HTMLElement {

    constructor() {
        super();
    }

    connectedCallback() {
        this.appendChild(document.querySelector("#color_organizer_template").content.cloneNode(true));

        this.querySelector("#organizer_addcolor").addEventListener("click", (event) => {
            const color_data = document.querySelector("color-picker").getVisiblePixels();
            this.addColor(color_data[0], color_data[1], color_data[2]);
        });
    }

    addColor = (color_lch, color_lch_sigma, color_rgb) => {
        const new_row = new ColorOrganizerRow(color_lch, color_lch_sigma, color_rgb);
        this.querySelector("tbody").appendChild(new_row.getDOMContent());
    }

};

class ColorOrganizerRow {
    dom_content;
    visible = true;

    color_id;
    color_name = "";
    color_lch = [];
    color_lch_sigma = [];
    color_rgb = [];

    constructor(color_lch, color_lch_sigma, color_rgb) {
        let color_data = document.querySelector("color-charts").addColor(color_lch, color_lch_sigma, color_rgb);
        this.color_id = color_data[0];
        this.color_lch = color_lch;
        this.color_lch_sigma = color_lch_sigma;
        this.color_rgb = color_rgb;
        this.color_name = color_data[1];

        // create row HTML from template
        this.dom_content = document.querySelector("#color_organizer_row_template").content.cloneNode(true);
        const data_classes = [".data_L", ".data_a", ".data_b"]
        for (let c = 0; c < 3; ++c) {
            this.dom_content.querySelector(data_classes[c]).innerHTML = String(color_lch[c].toFixed(2)) + " &#177; " + String(color_lch_sigma[c].toFixed(2));
            color_rgb[c] = parseInt(color_rgb[c]);
        }
        this.dom_content.querySelector(".color_name").innerText = this.color_name;
        this.dom_content.querySelector(".color_display").style["background-color"] = "rgb(" + color_rgb.join(",") + ")";

        // attach event handlers
        this.dom_content.querySelector(".toggle_visibility").addEventListener("click", this.toggleVisibility);
        this.dom_content.querySelector(".delete").addEventListener("click", this.removeRow);
        this.dom_content.querySelector(".color_name").addEventListener("input", this.updateName);
    }

    getDOMContent() {
        return this.dom_content;
    }

    removeRow = (event) => {
        document.querySelector("color-charts").hideColor(this.color_id);
        document.querySelector("color-organizer tbody").removeChild(event.target.parentElement.parentElement);
    }

    updateName = (event) => {
        this.color_name = event.target.innerText;
        document.querySelector("color-charts").showColor(this.color_id, this.color_lch, this.color_lch_sigma, this.color_rgb, this.color_name);
    }

    toggleVisibility = (event) => {
        if (this.visible) {
            document.querySelector("color-charts").hideColor(this.color_id);
            event.target.src = "./resources/visibility_off.png"
            this.visible = false;
        } else {
            document.querySelector("color-charts").showColor(this.color_id, this.color_lch, this.color_lch_sigma, this.color_rgb, this.color_name);
            event.target.src = "./resources/visibility.png"
            this.visible = true;
        }
    }
};
