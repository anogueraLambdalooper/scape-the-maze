import {Trainer} from "../../interfaces/Trainer.ts";
import {Perceptron} from "../Perceptron.ts";
import {Optimizer} from "../../interfaces/Optimizer.ts";
import {CanvasService} from "../../../application/services/CanvasService.ts";
import {LossFunction} from "../../interfaces/LossFunction.ts";

export class PerceptronTrainer implements Trainer {

    constructor(public optimizer: Optimizer){}

    public train(errorMargin: number, perceptron: Perceptron, lossFunction: LossFunction, traningDataset: number[], targetFunction: Function): void {
        let epochs = 0;
        let lossHistory = [];

        while (true) {
            epochs++;
            console.log("Epochs: " + epochs);
            for (const element of traningDataset) {
                let target = targetFunction(element);
                const output = perceptron.forward([element]);
                const error = target - output;

                lossHistory.push(lossFunction.evaluate(output, target));

                if (Math.abs(error) < errorMargin) {
                    console.info("Error", Math.abs(error));
                    console.info("WEIGHTS", perceptron.weights, perceptron.bias);
                    console.info("Solved!");
                    const canvas = new CanvasService();
                    canvas.printCanvas([], lossHistory, epochs, "Should arrive to the solution of y=2x");
                    return;
                }

                const adaptiveLearningRate = this.optimizer.updateLearningRate(epochs);

                if (epochs === 1) {
                    this.optimizer.initializeMomentum(perceptron.weights.length);
                }

                perceptron.backward([element], target, adaptiveLearningRate, this.optimizer);
            }
        }
    }
}