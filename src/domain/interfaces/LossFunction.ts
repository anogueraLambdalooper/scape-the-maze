export interface LossFunction {

    evaluate(output: number, target: number ) : number;

    derivative(output: number, target: number) : number;
}