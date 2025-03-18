import { ActivationFunction } from "./ActivationFunction.ts";
import {LossFunction} from "./LossFunction.ts";

export interface Trainer {
    train(errorMargin: number, objectToTrain: object, lossFunction: LossFunction, traningDataset: number[], targetFunction: Function, activationFunction: ActivationFunction): void;
}