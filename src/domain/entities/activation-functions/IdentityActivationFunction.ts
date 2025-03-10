import {ActivationFunction} from "../../interfaces/ActivationFunction.ts";

export class IdentityActivationFunction implements ActivationFunction {
    activate(input: number): number {
        return input;
    }
    derivative(_: number): number {
        return 1;
    }

}