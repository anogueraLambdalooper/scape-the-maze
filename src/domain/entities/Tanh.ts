import {ActivationFunction} from "../interfaces/ActivationFunction.ts";

export class Tanh implements ActivationFunction {

    activate(input: number): number {
        let output: number = 0;

        output = (Math.exp(input) - Math.exp(input * -1));
        output /= (Math.exp(input) + Math.exp(input * -1));

        return output;
    }

}