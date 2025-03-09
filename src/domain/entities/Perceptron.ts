import {ActivationFunction} from "../interfaces/ActivationFunction.ts";

export class Perceptron {
    private localGradient: number = 0;

    constructor(
        public weights: number[],
        public bias: number,
        public activationFunction: ActivationFunction,
        public learningRate: number
    ) {
    }

    forward(input: number[]): number {
        if (input.length !== this.weights.length) {
            throw new Error("Missmatch between inputs and weights length");
        }

        let output: number = 0;
        for (let i = 0; i < input.length; i++) {
            output += input[i] * this.weights[i];
        }
        output += this.bias;

        return this.activationFunction.activate(output);
    }

    backward(input: number[], target: number): void {
        if (input.length !== this.weights.length) {
            throw new Error("Missmatch between inputs and weights length");
        }

        //Output of the perceptron before activation.
        const preActivationOutput = input.reduce(
            (sum, val, i) => sum + val * this.weights[i],
            this.bias
        );

        //Error calculation
        let output: number = this.activationFunction.activate(preActivationOutput);
        let error: number = target - output;

        //Calculate localGradient
        this.localGradient = error * this.activationFunction.derivative(preActivationOutput);

        //Update the weights with this localGradient
        this.updateWeights(input);
    }

    getLocalGradient(): number {
        return this.localGradient;
    }

    private updateWeights(input: number[]): void {
        for (let i = 0; i < input.length; i++) {
            const gradient = this.learningRate * this.localGradient * input[i];
            this.weights[i] += gradient;
        }

        this.bias += this.learningRate * this.localGradient;
    }
}
