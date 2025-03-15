import {Optimizer} from "../interfaces/Optimizer.ts";

export class BatchGradientDescent implements Optimizer {
    optimize(initialLearningRate: number, drop: number, dropRate: number, iteration: number): number {
        return initialLearningRate * (Math.pow(drop, Math.floor((1 + iteration) / dropRate)));
    }

}