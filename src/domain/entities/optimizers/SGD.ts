import { Optimizer } from "../../interfaces/Optimizer.ts";
import { Perceptron } from "../Perceptron.ts";

export class SGD implements Optimizer {

    applyGradients(perceptron: Perceptron, learningRate: number, regularization: number, momentum: number): void {
        let weightDecay: number = (1 - regularization * learningRate);
        for (let i = 0; i < perceptron.weights.length; i++) {
            const weight = perceptron.weights[i];
            const weightVelocity = perceptron.weightVelocities[i] * momentum - perceptron.costGradientW[i] * learningRate;
            perceptron.weightVelocities[i] = weightVelocity;
            perceptron.weights[i] = weight * weightDecay + weightVelocity;
            perceptron.costGradientW[i] = 0;
        }

        const biasVelocity = perceptron.biasVelocity * momentum - perceptron.costGradientB * learningRate;
        perceptron.biasVelocity = biasVelocity;
        perceptron.bias += biasVelocity;
        perceptron.costGradientB = 0;
    }
}