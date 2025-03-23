export interface Scheduler {
    getLearningRate(epoch: number): number;
    getInitialLearningRate(): number;
}