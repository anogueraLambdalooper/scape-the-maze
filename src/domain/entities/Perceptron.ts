import {ActivationFunction} from "../interfaces/ActivationFunction.ts";

export class Perceptron {
    constructor(public weights: number[],
                public bias: number,
                public activationFunction: ActivationFunction,
                public learningRate: number) {}

    forward(input: number[]): number {
        if(input.length !== this.weights.length) {
            throw new Error("Missmatch between inputs and weights length");
        }

        let output: number = 0;
        for(let i = 0; i < input.length; i++) {
            output += input[i] * this.weights[i];
        }
        output += this.bias;

        return this.activationFunction.activate(output);
    }

    backward(input: number[], target: number): void {
        let preActivationOutput: number = 0;
        for(let i = 0; i < input.length; i++) {
            preActivationOutput += input[i] * this.weights[i];
        }

        preActivationOutput += this.bias;

        let output: number = this.activationFunction.activate(preActivationOutput);

        let error: number = target - output;

        let delta: number = error * this.activationFunction.derivative(preActivationOutput);

        for(let i = 0; i < input.length; i++) {
            this.weights[i] += this.learningRate * delta * input[i];
        }

        this.bias += this.learningRate * delta;
    }
}