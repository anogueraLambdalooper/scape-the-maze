import { ActivationFunction } from "./ActivationFunction.ts";
import {PerceptronLossFunction} from "./PerceptronLossFunction.ts";

export interface Trainer {
    train(errorMargin: number, objectToTrain: object, lossFunction: PerceptronLossFunction, traningDataset: number[], targetFunction: Function, activationFunction: ActivationFunction): void;
}