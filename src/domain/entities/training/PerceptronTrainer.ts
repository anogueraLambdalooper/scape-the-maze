import {Trainer} from "../../interfaces/Trainer.ts";
import {Perceptron} from "../Perceptron.ts";
import {Optimizer} from "../../interfaces/Optimizer.ts";
import {CanvasService} from "../../../application/services/CanvasService.ts";
import {LossFunction} from "../../interfaces/LossFunction.ts";
import { ActivationFunction } from "../../interfaces/ActivationFunction.ts";

export class PerceptronTrainer implements Trainer {

    constructor(public optimizer: Optimizer){}

    public train(errorMargin: number, perceptron: Perceptron, lossFunction: LossFunction, traningDataset: number[], targetFunction: Function, activationFunction: ActivationFunction): boolean {
        let epochs = 0;
        let lossHistory = [];

        while (true) {
            epochs++;
            console.log("Epochs: " + epochs);
            for (const element of traningDataset) {

                //guide
                //output = model.forward(text, offsets)
                //loss = criterion(output, label)
                //loss.backward()
                //optimizer.step()

                //train_loss += loss.item() * len(output)
                //train_acc += (output.argmax(1) == label).sum().item()

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

                const adaptiveLearningRate = this.optimizer.updateLearningRate(epochs);

                if (epochs === 1) {
                    this.optimizer.initializeMomentum(perceptron.weights.length);
                }

                const errorGradient = perceptron.backward(target, output);
                const gradient = errorGradient * activationFunction.derivative(preActivationOutput);
                const weightGradients = element * gradient;
                this.optimizer.update(perceptron, weightGradients, gradient, adaptiveLearningRate);
            }
        }
    }
}