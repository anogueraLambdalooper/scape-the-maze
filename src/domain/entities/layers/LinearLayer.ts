import {Layer} from "../../interfaces/Layer.ts";

export class LinearLayer implements Layer {
    public weights: number[][];
    public biases: number[];

    constructor(weights: number[][], biases: number[]) {
        this.weights = weights;
        this.biases = biases;
    }

    forwardPass(input: number[]): number[] {
        return this.weights.map((row, i) => {
            return row.reduce((acc, weight, j) => acc + weight * input[j], this.biases[i]);
        });
    }

    backwardPass(inputs: number[], outputs: number[], targets: Function): number[] {
        let gradients: number[] = [];
        for(let i = 0; i < outputs.length; i++) {
            gradients.push(outputs[i] - targets(inputs[i]));
        }

        return gradients;
    }
}