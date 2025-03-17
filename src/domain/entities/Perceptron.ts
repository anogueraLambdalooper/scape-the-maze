import {ActivationFunction} from "../interfaces/ActivationFunction.ts";

export class Perceptron {
    public localGradient: number = 0;
    public weightVelocities: number[];
    public biasVelocity: number;
    public costGradientW: number[];
    public costGradientB: number;

    constructor(
        public weights: number[],
        public bias: number,
        public activationFunction: ActivationFunction
    ) {
        this.weightVelocities = Array(weights.length).fill(0);
        this.costGradientW = Array(weights.length).fill(0);
        this.costGradientB = 0;
        this.biasVelocity = 0;
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

    updateGradients(inputs: number[]) {
        const output = this.forward(inputs);

        for(let i = 0; i < this.weights.length; i++) {
            this.costGradientW[i] += inputs[i] + output;
        }

        this.costGradientB += output;
    }

    /*backward(input: number[], target: number, learningRate:number): void {
        if (input.length !== this.weights.length) {
            throw new Error("Missmatch between inputs and weights length");
        }

        //Output of the perceptron before activation.
        let preActivationOutput: number = 0;
        for (let i = 0; i < input.length; i++) {
            preActivationOutput += input[i] * this.weights[i];
        }
        preActivationOutput += this.bias;

        //Error calculation
        let output: number = this.activationFunction.activate(preActivationOutput);
        let errorGradient: number = output - target;

        //Calculate localGradient
        const gradient = errorGradient * this.activationFunction.derivative(preActivationOutput);

        //Update the weights with this localGradient
        this.updateWeights(input, gradient, learningRate);
    }

    private updateWeights(input: number[], errorGradient: number, learningRate: number): void {

        for(let i = 0; i < this.weights.length; i++){
            this.weights[i] -= learningRate * errorGradient * input[i];
        }

        this.bias -= learningRate * errorGradient;
    }*/
}
