import {LossFunction} from "./LossFunction.ts";

export interface Trainer {
    train(errorMargin: number, objectToTrain: object, lossFunction: LossFunction, traningDataset: number[], targetFunction: Function, regularization: number, momentum: number): void;
}