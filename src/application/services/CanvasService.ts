import {Chart, ChartConfiguration, registerables} from "chart.js";
import {Canvas, createCanvas} from "canvas";
import * as fs from "fs";

Chart.register(...registerables);

export class CanvasService {
    private width: number = 800;
    private height: number = 600;
    private canvas: Canvas;

    constructor() {
        this.canvas = createCanvas(this.width, this.height);
    }

    public printCanvas(errors: number[], epochs: number, canvasName: string) {
        const ctx = this.canvas.getContext("2d");

        if (!ctx) {
            throw new Error("No se pudo obtener el contexto del canvas");
        }

        const epochsArray = Array.from({length: epochs}, (_, i) => i);

        const config: ChartConfiguration = {
            type: "line",
            data: {
                labels: epochsArray.map(String),
                datasets: [
                    {
                        label: "Error vs Época",
                        data: errors,
                        borderColor: "red",
                        backgroundColor: "rgba(255, 99, 132, 0.2)",
                        tension: 0.3,
                    },
                ],
            },
            options: {
                responsive: false,
                plugins: {
                    legend: {display: true},
                    title: {display: true, text: "Error Evolution"},
                },
                scales: {
                    x: {title: {display: true, text: "Epoch"}},
                    y: {title: {display: true, text: "Error"}},
                },
            },
        };

        new Chart(ctx as any, config);

        const buffer = this.canvas.toBuffer("image/png");
        fs.writeFileSync("./canvas/"+canvasName + ".png", buffer);

        console.log("✅ Imagen generada: " + canvasName + ".png");
    }

}