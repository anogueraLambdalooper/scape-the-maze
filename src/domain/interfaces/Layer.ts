export interface Layer {
    weights: number[][];
    biases: number[];
    forwardPass(input: number[]): number[];
    backwardPass(inputs: number[], outputs: number[], targets: Function): number[];
}