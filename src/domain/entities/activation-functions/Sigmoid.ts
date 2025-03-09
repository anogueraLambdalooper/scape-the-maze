import { ActivationFunction } from "../../interfaces/ActivationFunction.ts";

export class Sigmoid implements ActivationFunction {
  activate(input: number): number {
    let output: number = 0;

    output = 1 / (1 + Math.exp(input * -1));

    return output;
  }

  derivative(input: number): number {
    // sig'(x) = sig * (1-sig)
    const sig = this.activate(input);
    return sig * (1 - sig);
  }
}
