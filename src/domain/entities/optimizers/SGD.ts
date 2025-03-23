import { Optimizer } from "../../interfaces/Optimizer.ts";

export class SGD implements Optimizer {
    private learningRate: number;
    private momentum: number;
    private velocities: number[];
    private biasVelocity: number;

    constructor(learningRate: number, momentum: number = 0.9) {
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.velocities = [];
        this.biasVelocity = 0;
    }

    public updateParameters(parameters: number[], gradients: number[]): void {
        if (this.velocities.length === 0) {
            this.velocities = new Array(parameters.length).fill(0);
        }

        for (let i = 0; i < parameters.length; i++) {
            this.velocities[i] = this.momentum * this.velocities[i] - this.learningRate * gradients[i];
            parameters[i] += this.velocities[i];
        }
    }

    public updateBias(bias: number, gradient: number): number {
        this.biasVelocity = this.momentum * this.biasVelocity - this.learningRate * gradient;
        return bias + this.biasVelocity;
    }

    public setLearningRate(learningRate: number): void {
        this.learningRate = learningRate;
    }

    public getLearningRate(): number {
        return this.learningRate;
    }
}