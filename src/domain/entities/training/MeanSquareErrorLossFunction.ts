import {LossFunction} from "../../interfaces/LossFunction.ts";

export class MeanSquareErrorLossFunction implements LossFunction{

    public evaluate(output: number, target: number ): number {
        return ((output - target) * (output - target)) / 2;
    }

    public derivative(output: number, target: number): number {
        return output - target;
    }
}