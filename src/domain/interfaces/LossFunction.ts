export interface LossFunction {

    evaluate(output: number, target: number ) : number;
}