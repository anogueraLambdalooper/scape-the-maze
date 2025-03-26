import {LayerOptimizer} from "../../interfaces/LayerOptimizer.ts";

export class LayerSGD implements LayerOptimizer {
    private learningRate: number;
    private momentum: number;
    private velocities: number[][];
    private biasVelocity: number;

    constructor(learningRate: number, momentum: number = 0.9) {
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.velocities = [[]];
        this.biasVelocity = 0;
    }

    initialize(parameters: number[][]): void {
        this.velocities = new Array(parameters.length).fill(0);
        for (let i = 0; i < parameters.length; i++) {
            this.velocities[i] = new Array(parameters[i].length).fill(0);
        }
    }

    public updateParameters(parameters: number[][], gradients: number[][]): void {
        for (let i = 0; i < parameters.length; i++) {
            for (let j = 0; j < parameters[i].length; j++) {
                this.velocities[i][j] = this.momentum * this.velocities[i][j] - this.learningRate * gradients[i][j];
                parameters[i][j] += this.velocities[i][j];
            }
        }
    }

    public updateBias(bias: number[], gradient: number[]): void {
        for (let i = 0; i < bias.length; i++) {
            this.biasVelocity = this.momentum * this.biasVelocity - this.learningRate * gradient[i];
            bias[i] += this.biasVelocity;
        }
    }

    public setLearningRate(learningRate: number): void {
        this.learningRate = learningRate;
    }

    public getLearningRate(): number {
        return this.learningRate;
    }
}