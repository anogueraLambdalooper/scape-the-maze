import {Trainer} from "../../interfaces/Trainer.ts";
import {Perceptron} from "../Perceptron.ts";
import {Optimizer} from "../../interfaces/Optimizer.ts";
import {LossFunction} from "../../interfaces/LossFunction.ts";
import {CanvasService} from "../../../application/services/CanvasService.ts";

export class PerceptronTrainer implements Trainer {

    constructor(public optimizer: Optimizer){}

    public train(learningRate: number, errorMargin: number, perceptron: Perceptron, lossFunction: LossFunction, traningDataset: number[], targetFunction: Function): void {
        let epochs = 0;
        let lossHistory = [];

        while(true) {
             for(const element of traningDataset) {
                 let target = targetFunction(element);
                 const output = perceptron.forward([element])
                 const error = target - output;

                 epochs++;
                 lossHistory.push(lossFunction.evaluate(output, target));

                 if(Math.abs(error) < errorMargin) {
                     console.info("Error", Math.abs(error));
                     console.info("WEIGHTS", perceptron.weights, perceptron.bias);
                     console.info("Solved!");
                     const canvas = new CanvasService();
                     canvas.printCanvas([], lossHistory, epochs, "Should arrive to the solution of y=2x");
                     return;
                 }

                 perceptron.backward([element], target, this.optimizer.optimize(learningRate, 0.5, 100, epochs));
             }
        }
    }
}