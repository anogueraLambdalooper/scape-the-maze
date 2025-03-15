export interface Optimizer {
    optimize(initialLearningRate: number, drop: number, dropRate: number, iteration: number): number;
}