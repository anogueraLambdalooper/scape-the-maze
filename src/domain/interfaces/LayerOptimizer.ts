export interface LayerOptimizer {
    initialize(parameters: number[][]): void;
    updateParameters(parameters: number[][], gradients: number[][]): void;
    updateBias(bias: number[], gradient: number[]): void;
    setLearningRate(learningRate: number): void;
    getLearningRate(): number;
}