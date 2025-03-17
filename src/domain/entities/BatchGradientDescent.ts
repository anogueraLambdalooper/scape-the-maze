export class BatchGradientDescent {
    optimize(initialLearningRate: number, drop: number, dropRate: number, iteration: number): number {
        return initialLearningRate * (Math.pow(drop, Math.floor((1 + iteration) / dropRate)));
    }

}