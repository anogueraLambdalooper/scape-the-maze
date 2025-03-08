export interface ActivationFunction {
    activate(input: number): number;
    derivative(x: number): number;
}