export interface PerceptronOptimizer {
    updateParameters(parameters: number[], gradients: number[]): void;
    updateBias(bias: number, gradient: number): number;
    setLearningRate(learningRate: number): void;
    getLearningRate(): number;
}