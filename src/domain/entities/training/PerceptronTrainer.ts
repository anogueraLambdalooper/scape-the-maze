import {Trainer} from "../../interfaces/Trainer.ts";
import {Perceptron} from "../Perceptron.ts";
import {Optimizer} from "../../interfaces/Optimizer.ts";
import {CanvasService} from "../../../application/services/CanvasService.ts";
import {LossFunction} from "../../interfaces/LossFunction.ts";
import { ActivationFunction } from "../../interfaces/ActivationFunction.ts";
import {Scheduler} from "../../interfaces/Scheduler.ts";

export class PerceptronTrainer implements Trainer {

    constructor(public optimizer: Optimizer, public scheduler: Scheduler){}

    public train(errorMargin: number, perceptron: Perceptron, lossFunction: LossFunction, traningDataset: number[], targetFunction: Function, activationFunction: ActivationFunction): boolean {
        let epochs = 0;
        let lossHistory = [];

        while (true) {
            epochs++;
            for (const element of traningDataset) {
                let target = targetFunction(element);

                /*PERCEPTRON FORWARD*/
                const preActivationOutput = perceptron.forward([element]);
                const output = activationFunction.activate(preActivationOutput);
                /*******************/

                /*loss from a criterion (in this case the lossFunction.evaluate()*/
                const loss = lossFunction.evaluate(output, target);

                /*CHECK CONDITION TO CLOSE AND ADD TO CHART*/
                lossHistory.push(loss);
                if (Math.abs(loss) < errorMargin) {
                    console.info("Error", Math.abs(loss));
                    console.info("WEIGHTS", perceptron.weights, perceptron.bias);
                    console.info("Solved!");
                    const canvas = new CanvasService();
                    canvas.printCanvas([], lossHistory, epochs, "Should arrive to the solution of y=2x");
                    return true;
                }
                /*****************************************/

                /*equivalent to loss.backward()*/
                const errorGradient = perceptron.backward(target, output);
                /*******************************/

                /*Equivalent to optimizer.step()*/
                const gradient = errorGradient * activationFunction.derivative(preActivationOutput);
                const weightGradients = element * gradient;

                const currentLearningRate = this.scheduler.getLearningRate(epochs);
                this.optimizer.setLearningRate(currentLearningRate);
                this.optimizer.updateParameters(perceptron.weights, [weightGradients]);
                perceptron.bias = this.optimizer.updateBias(perceptron.bias, gradient);
                console.log("WEIGHTS", perceptron.weights, perceptron.bias);
                /*******************************/
            }
        }
    }
}