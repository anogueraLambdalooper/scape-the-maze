import {Layer} from "../../interfaces/Layer.ts";
import {Perceptron} from "../Perceptron.ts";
import {ActivationFunction} from "../../interfaces/ActivationFunction.ts";

export class Linear implements Layer {
    public perceptrons: Perceptron[];
    private lastInput: number[] = [];

    constructor(in_features: number,
                out_features: number,
                activationFunction: ActivationFunction,
                bias: number = 1,
                learningRate: number = 0.1) {

        this.perceptrons = Array.from({length: out_features}, () => {
            const weights = Array(in_features).fill(0).map(() =>
                Math.random() * 2 - 1
            );
            return new Perceptron(weights, bias, activationFunction, learningRate);
        });
    }

    forwardPass(input: number[]): number[] {
        this.lastInput = [...input];
        return this.perceptrons.map((p) => p.forward(input));
    }

    backwardPass(targets: number[]): number[] {
        if (targets.length !== this.perceptrons.length) {
            throw new Error("Missmatch between amount of targets and perceptrons");
        }

        //Get Gradient of each Perceptron
        const gradients = this.perceptrons.map((perceptron, i) => {
            const preActivation = perceptron.weights.reduce((sum, weight, j) =>
                sum + weight * this.lastInput[j], perceptron.bias
            );
            const output = perceptron.activationFunction.activate(preActivation);
            const error = targets[i] - output;
            return error * perceptron.activationFunction.derivative(preActivation);
        });

        //Get Gradients for the input perceptrons of the next layer in the backward order
        const inputGradients = this.lastInput.map((_, i) =>
            this.perceptrons.reduce((sum, perceptron) => sum + perceptron.weights[i] * gradients[perceptron.weights[i]], 0)
        );

        // 3. Update weights and bias for each perceptron in layer
        this.perceptrons.forEach((perceptron, i) => {
            perceptron.backward(this.lastInput, targets[i]);
        });

        return inputGradients;
    }
}