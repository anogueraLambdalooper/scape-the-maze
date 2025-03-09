export interface Layer {
    forwardPass(input: number[]): number[];
    backwardPass(targets: number[]): number[];
}