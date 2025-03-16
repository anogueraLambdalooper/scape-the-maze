import { CostIdentity } from "../../interfaces/CostIdentity";

export class CrossEntropy implements CostIdentity {

    costFunction(outputs: number[], targets: number[]): number {
        let cost: number = 0;

        for(let i = 0; i < outputs.length; i++) {
            let x = outputs[i];
            let y = targets[i];
            let v = (y === 1) ? -Math.log(x) : -Math.log(1 - x);
            cost += isNaN(v) ? 0 : v;
        }

        return cost;
    }

    costDerivative(output: number, target: number): number {
        let x = output;
        let y = target;

        if (x === 0 || x === 1) {
            return 0;
        }
        
        return (-x + y) / (x * (x - 1));
    }

}