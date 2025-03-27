import {LCHtoRGB} from "./tools.js";
import {color_key} from "./color-key.js";

const sortby_L = (x, y) => x[5] - y[5];
const sortby_a = (x, y) => x[6] - y[6];
const sortby_b = (x, y) => x[7] - y[7];

export default class ColorReference extends HTMLElement {

    constructor() {
        super();
    }

    connectedCallback() {
        const clone = document.getElementById("color_reference_template").content.cloneNode(true);
        this.appendChild(clone);

        this.addAllColors(sortby_a);
    }

    addAllColors(sorting) {
        this.querySelector("tbody").innerHTML = "";
        const data_classes = [".data_L", ".data_a", ".data_b"]

        color_key.sort(sorting).forEach((ref) => {
            const clone = document.getElementById("color_reference_row_template").content.cloneNode(true);
            for (let c = 0; c < 3; ++c) {
                clone.querySelector(data_classes[c]).innerHTML = String(ref[c+5].toFixed(2));
            }
            clone.querySelector(".color_name").innerText = ref[0];
            clone.querySelector(".color_display").style["background-color"] = ref[1];

            this.querySelector("tbody").appendChild(clone);
        });
    }

};
