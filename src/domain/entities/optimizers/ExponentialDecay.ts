import {Scheduler} from "../../interfaces/Scheduler.ts";

export class ExponentialDecay implements Scheduler {
    private initialLearningRate: number;
    private decayRate: number;

    constructor(initialLearningRate: number, decayRate: number) {
        this.initialLearningRate = initialLearningRate;
        this.decayRate = decayRate;
    }

    public getLearningRate(epoch: number): number {
        return this.initialLearningRate * Math.pow(this.decayRate, epoch);
    }

    public getInitialLearningRate(): number {
        return this.initialLearningRate;
    }
}