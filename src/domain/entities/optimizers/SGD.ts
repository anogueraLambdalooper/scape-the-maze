import { Optimizer } from "../../interfaces/Optimizer.ts";
import {Perceptron} from "../Perceptron.ts";

export class SGD implements Optimizer {
    private momentumVelocity: number[];
    private biasVelocity: number;

    constructor(public initialLearningRate: number, public decayRate: number = 0.001, public minLearningRate: number = 0.000001, public momentum: number = 0.9) {
        this.momentumVelocity = [];
        this.biasVelocity = 0;
    }

    updateLearningRate(epoch: number): number {
        let newLearningRate = this.initialLearningRate / (1 + this.decayRate * epoch);
        return Math.max(newLearningRate, this.minLearningRate);
    }

    initializeMomentum(weightsLength: number): void {
        if (this.momentumVelocity.length === 0) {
            this.momentumVelocity = Array(weightsLength).fill(0);
            this.biasVelocity = 0;
        }
    }

    update(perceptron: Perceptron, gradient: number[], gradientB: number, learningRate: number): void {
        if (this.momentumVelocity.length === 0) return;

        for (let i = 0; i < perceptron.weights.length; i++) {
            this.momentumVelocity[i] = this.momentum * this.momentumVelocity[i] + gradient[i];
            perceptron.weights[i] -= learningRate * this.momentumVelocity[i];
        }

        this.biasVelocity = this.momentum * this.biasVelocity + gradientB;
        perceptron.bias -= learningRate * this.biasVelocity;
    }
}