import {RGBtoLCH, stdev} from "./tools.js";

export default class ColorPicker extends HTMLElement {

    // DOM elements
    image;
    canvas;
    ctx;

    // zoom/pan state variables
    scale = 1;
    minScale = 1;
    maxScale = 100;
    imgX = 0;
    imgY = 0;
    isDragging = false;
    lastMouseX = 0;
    lastMouseY = 0;

    // picker variables
    rectXpct = 25;
    rectYpct = 25;
    lineWidth = 5;

    constructor() {
        super();
    }

    connectedCallback() {
        const template = document.getElementById("color_picker_template");
        const clone = template.content.cloneNode(true);
        this.appendChild(clone);

        const file_input = this.querySelector("#picker_file");
        file_input.addEventListener("change", this.imageInputChanged);

        const file_range_min = this.querySelector("#picker_range_min");
        file_range_min.addEventListener("input", this.rangeMinChanged);

        const file_range_max = this.querySelector("#picker_range_max");
        file_range_max.addEventListener("input", this.rangeMaxChanged);

        this.image = new Image();
        this.image.addEventListener("load", this.imageLoaded);

        this.canvas = this.querySelector("#image_canvas");
        this.canvas.addEventListener("wheel", this.canvasWheel);
        this.canvas.addEventListener("mousedown", this.canvasMouseDown);
        this.canvas.addEventListener("mousemove", this.canvasMouseMove);
        this.canvas.addEventListener("mouseup", this.stopPanning);
        this.canvas.addEventListener("mouseleave", this.stopPanning);

        this.ctx = this.canvas.getContext("2d");

    }

    imageInputChanged = (event) => {
        const file = event.target.files[0];
        if (file && file.type.startsWith("image/")) {
            const reader = new FileReader();
            reader.onload = (e) => { this.image.src = e.target.result; };
            reader.readAsDataURL(file);
        } else {
            alert("Please upload a valid image file.");
        }
    }

    rangeMinChanged = (event) => { this.rectXpct = event.target.value; this.drawImage(); }
    rangeMaxChanged = (event) => { this.rectYpct = event.target.value; this.drawImage(); }

    imageLoaded = () => {
        const aspectRatio = this.image.width / this.image.height;
        this.canvas.height = window.innerHeight - 100;
        this.canvas.width = this.canvas.height * aspectRatio;

        this.scale = this.canvas.width / this.image.width;
        this.minScale = this.scale;
        this.imgX = 0;
        this.imgY = 0;

        this.drawImage();
    }

    canvasWheel = (event) => {
        event.preventDefault();

        const rect = this.canvas.getBoundingClientRect();
        const mouseX = event.clientX - rect.left;
        const mouseY = event.clientY - rect.top;

        const zoomFactor = event.deltaY > 0 ? 0.9 : 1.1;
        let newScale = this.scale * zoomFactor;
        if (newScale < this.minScale || newScale > this.maxScale) return;

        const dx = (mouseX - this.imgX) / this.scale;
        const dy = (mouseY - this.imgY) / this.scale;

        this.imgX = mouseX - dx * newScale;
        this.imgY = mouseY - dy * newScale;
        this.scale = newScale;

        this.drawImage();
    }

    canvasMouseDown = (event) => {
        this.isDragging = true;
        this.lastMouseX = event.clientX;
        this.lastMouseY = event.clientY;
        this.canvas.style.cursor = "grabbing";
    }

    canvasMouseMove = (event) => {
        if (!this.isDragging) return;
        let dx = event.clientX - this.lastMouseX;
        let dy = event.clientY - this.lastMouseY;
        this.imgX += dx;
        this.imgY += dy;
        this.lastMouseX = event.clientX;
        this.lastMouseY = event.clientY;

        this.drawImage();
    }

    stopPanning = () => {
        this.isDragging = false;
        this.canvas.style.cursor = "grab";
    }

    drawImage() {
        // keep image within canvas bounds
        const scaledWidth = this.image.width * this.scale;
        const scaledHeight = this.image.height * this.scale;
        this.imgX = Math.min(0, Math.max(this.canvas.width - scaledWidth, this.imgX));
        this.imgY = Math.min(0, Math.max(this.canvas.height - scaledHeight, this.imgY));

        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.drawImage(this.image, this.imgX, this.imgY, this.image.width * this.scale, this.image.height * this.scale);

        // draw rectangle on top of image
        this.ctx.beginPath();
        this.ctx.lineWidth = this.lineWidth;
        this.ctx.strokeStyle = "orange";
        const rectWidth = parseInt(this.rectXpct * this.canvas.width / 100);
        const rectHeight = parseInt(this.rectYpct * this.canvas.height / 100);
        const rectX = parseInt((this.canvas.width - rectWidth)/2);
        const rectY = parseInt((this.canvas.height - rectHeight)/2);
        this.ctx.rect(rectX, rectY, rectWidth, rectHeight);
        this.ctx.stroke();
    }

    getVisiblePixels() {
        const rectWidth = parseInt(this.rectXpct * this.canvas.width / 100) - 2*this.lineWidth;
        const rectHeight = parseInt(this.rectYpct * this.canvas.height / 100) - 2*this.lineWidth;
        const rectX = parseInt((this.canvas.width - rectWidth)/2);
        const rectY = parseInt((this.canvas.height - rectHeight)/2);
        const imageData = this.ctx.getImageData(rectX, rectY, rectWidth, rectHeight);
        const pixels = imageData.data;
        const lch_l = [];
        const lch_c = [];
        const lch_h = [];
        const rgb_r = [];
        const rgb_g = [];
        const rgb_b = [];
        for (let i = 0; i < pixels.length; i += 4) {
            let lch = RGBtoLCH([pixels[i], pixels[i+1], pixels[i+2]]);
            lch_l.push(lch[0]);
            lch_c.push(lch[1]);
            lch_h.push(lch[2]);
            rgb_r.push(pixels[i]);
            rgb_g.push(pixels[i+1]);
            rgb_b.push(pixels[i+2]);
        }
        const lch_l_s = stdev(lch_l);
        const lch_c_s = stdev(lch_c);
        const lch_h_s = stdev(lch_h);
        const rgb_r_s = stdev(rgb_r);
        const rgb_g_s = stdev(rgb_g);
        const rgb_b_s = stdev(rgb_b);
        return [[lch_l_s[0], lch_c_s[0], lch_h_s[0]], [lch_l_s[1], lch_c_s[1], lch_h_s[1]], [rgb_r_s[0], rgb_g_s[0], rgb_b_s[0]]];
    }
};
