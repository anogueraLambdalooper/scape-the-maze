import {Perceptron} from "../entities/Perceptron.ts";

export interface Optimizer {
    applyGradients(perceptrons: Perceptron[], learningRate: number, regularization: number, momentum: number): void;
}