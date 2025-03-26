export interface LayerLossFunction {

    evaluate(inputs: number[], output: number[], target: Function) : number;
}