export class Perceptron {

    constructor(
        public weights: number[],
        public bias: number
    ) {}

    forward(input: number[]): number {
        if (input.length !== this.weights.length) {
            throw new Error("Missmatch between inputs and weights length");
        }

        let output: number = 0;
        for (let i = 0; i < input.length; i++) {
            output += input[i] * this.weights[i];
        }

        return output += this.bias;
    }

    backward(target: number, output:number): number {
        return output - target;
    }
}
