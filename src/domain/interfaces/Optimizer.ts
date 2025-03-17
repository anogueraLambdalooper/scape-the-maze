import {Perceptron} from "../entities/Perceptron.ts";

export interface Optimizer {
    applyGradients(perceptron: Perceptron, learningRate: number, regularization: number, momentum: number): void;
}