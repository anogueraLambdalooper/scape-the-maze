import {ActivationFunction} from "../interfaces/ActivationFunction.ts";
import {Optimizer} from "../interfaces/Optimizer.ts";

export class Perceptron {
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

    backward(input: number[], target: number, learningRate:number, optimizer: Optimizer): void {
        if (input.length !== this.weights.length) {
            throw new Error("Missmatch between inputs and weights length");
        }

        let preActivationOutput: number = 0;
        for (let i = 0; i < input.length; i++) {
            preActivationOutput += input[i] * this.weights[i];
        }
        preActivationOutput += this.bias;

        let output: number = this.activationFunction.activate(preActivationOutput);
        let errorGradient: number = output - target;

        const gradient = errorGradient * this.activationFunction.derivative(preActivationOutput);
        const weightGradients = input.map(x => x * gradient);
        optimizer.update(this, weightGradients, gradient, learningRate);
    }

    /*private updateWeights(input: number[], errorGradient: number, learningRate: number): void {

        for(let i = 0; i < this.weights.length; i++){
            this.weights[i] -= learningRate * errorGradient * input[i];
        }

        this.bias -= learningRate * errorGradient;
    }*/

    //this.updateGradients(input, output);
    //optimizer.applyGradients(this, learningRate / datasetLength, regularization, momentum);
    //this.clearAllGradients();
    /*updateGradients(input: number[], output: number) {

        for(let i = 0; i < this.weights.length; i++) {
            this.costGradientW[i] += input[i] + output;
        }

        this.costGradientB += output;
    }

    clearAllGradients() {
        for(let i = 0; i < this.costGradientW.length; i++) {
            this.costGradientW[i] = 0;
        }
        this.costGradientB = 0;
    }*/
}
