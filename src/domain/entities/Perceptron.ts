import {ActivationFunction} from "../interfaces/ActivationFunction.ts";

export class Perceptron {
    constructor(public weights: number[],
                public bias: number,
                public activationFunction: ActivationFunction) {}

    forward(input: number[]): number {
        if(input.length !== this.weights.length) {
            throw new Error("Missmatch between inputs and weights length");
        }

        let output: number = 0;
        for(let i = 0; i < input.length; i++) {
            output = input[i] * this.weights[i];
        }
        output += this.bias;

        return this.activationFunction.activate(output);
    }
}