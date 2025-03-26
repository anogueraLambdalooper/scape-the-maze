import { ActivationFunction } from "./ActivationFunction.ts";
import {Layer} from "./Layer.ts";
import {LayerLossFunction} from "./LayerLossFunction.ts";

export interface LTrainer {
    train(errorMargin: number, layer: Layer, lossFunction: LayerLossFunction, traningDataset: number[], targetFunction: Function, activationFunction: ActivationFunction): boolean;
}