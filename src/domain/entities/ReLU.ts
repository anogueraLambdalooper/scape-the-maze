﻿import {ActivationFunction} from "../interfaces/ActivationFunction.ts";

export class ReLU implements ActivationFunction {
    activate(input: number): number {
        let output: number = 0;

        output = Math.max(0, input)

        return output;
    }

}