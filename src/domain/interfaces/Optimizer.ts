import {Perceptron} from "../entities/Perceptron.ts";

export interface Optimizer {
    updateLearningRate(epoch: number): number;
    initializeMomentum(weightsLength: number): void;
    update(perceptron: Perceptron, gradient: number[], gradientB: number, learningRate: number): void;
}