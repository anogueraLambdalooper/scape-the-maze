import {LossFunction} from "./LossFunction.ts";

export interface Trainer {
    train(learningRate: number, errorMargin: number, objectToTrain: object, lossFunction: LossFunction, traningDataset: number[], targetFunction: Function): void;
}