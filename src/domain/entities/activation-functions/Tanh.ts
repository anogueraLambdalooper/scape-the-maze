import { ActivationFunction } from "../../interfaces/ActivationFunction.ts";

export class Tanh implements ActivationFunction {
  activate(input: number): number {
    const ex = Math.exp(input);
    const e_x = Math.exp(-input);

    return (ex - e_x) / (ex + e_x);
  }

  derivative(input: number): number {
    // tanh'(x) = 1 - tanh²(x)
    const tanhValue = this.activate(input);
    return 1 - tanhValue * tanhValue;
  }
}
