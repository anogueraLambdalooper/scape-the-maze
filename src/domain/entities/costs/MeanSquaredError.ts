import { CostIdentity } from "../../interfaces/CostIdentity";

export class MeanSquaredError implements CostIdentity {
    costFunction(outputs: number[], targets: number[]): number {
        
        let cost: number = 0;

        for(let i = 0; i < outputs.length; i++) {
            let error = outputs[i] - targets[i];
            cost += error * error;
        }

        return 0.5 * cost;        
    }

    costDerivative(output: number, target: number): number {
        return target - output;
    }

}