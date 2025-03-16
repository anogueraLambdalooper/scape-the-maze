export interface CostIdentity {
    costFunction(outputs: number[], targets: number[]): number;

    costDerivative(output: number, target: number): number;
}