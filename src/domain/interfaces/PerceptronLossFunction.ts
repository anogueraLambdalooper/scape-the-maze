export interface PerceptronLossFunction {

    evaluate(output: number, target: number ) : number;
}