import {LayerLossFunction} from "../../interfaces/LayerLossFunction.ts";


export class LayerMeanSquareErrorLossFunction implements LayerLossFunction {

    public evaluate(inputs: number[], output: number[], target: Function): number {
        let error = 0;
        for (let i = 0; i < output.length; i++) {
            error += ((output[i] - target(inputs[i])) * (output[i] - target(inputs[i]))) / 2;
        }
        return error;
    }
}