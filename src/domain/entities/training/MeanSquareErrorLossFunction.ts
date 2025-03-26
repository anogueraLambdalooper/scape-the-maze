import {PerceptronLossFunction} from "../../interfaces/PerceptronLossFunction.ts";

export class MeanSquareErrorLossFunction implements PerceptronLossFunction{

    public evaluate(output: number, target: number ): number {
        return ((output - target) * (output - target)) / 2;
    }
}