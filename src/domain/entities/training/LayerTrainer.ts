import {ActivationFunction} from "../../interfaces/ActivationFunction.ts";
import {Layer} from "../../interfaces/Layer.ts";
import {Scheduler} from "../../interfaces/Scheduler.ts";
import {LayerLossFunction} from "../../interfaces/LayerLossFunction.ts";
import {LayerOptimizer} from "../../interfaces/LayerOptimizer.ts";
import {LTrainer} from "../../interfaces/LTrainer.ts";
import {CanvasService} from "../../../application/services/CanvasService.ts";

export class LayerTrainer implements LTrainer {

    constructor(public optimizer: LayerOptimizer, public scheduler: Scheduler) {
    }

    train(errorMargin: number, layer: Layer, lossFunction: LayerLossFunction, traningDataset: number[], targetFunction: Function, activationFunction: ActivationFunction): boolean {
        let epochs = 0;
        let lossHistory = [];

        while (true) {
            epochs++;

            if(epochs === 1) {
                this.optimizer.initialize(layer.weights);
            }

            /*LAYER FORWARD*/
            const preActivationOutput = layer.forwardPass(traningDataset);
            const output = preActivationOutput.map(activationFunction.activate);
            /*******************/

            /*loss from a criterion (in this case the lossFunction.evaluate()*/
            const loss = lossFunction.evaluate(traningDataset, output, targetFunction);
            lossHistory.push(loss);

            /*CHECK CONDITION TO CLOSE AND ADD TO CHART*/
            if (Math.abs(loss) < errorMargin) {
                console.info("Error", Math.abs(loss));
                console.info("WEIGHTS", layer.weights, layer.biases);
                console.info("Solved!");
                const canvas = new CanvasService();
                canvas.printCanvas([], lossHistory, epochs, "Training Loss Epochs on Layer");
                return true;
            }
            /*****************************************/

            /*equivalent to loss.backward()*/
            const errorGradients = layer.backwardPass(traningDataset, output, targetFunction);
            /*******************************/

            /*Equivalent to optimizer.step()*/
            const gradients = errorGradients.map((_, i) => errorGradients[i] * activationFunction.derivative(preActivationOutput[i]));
            const weightGradients = layer.weights.map((row, i) => row.map((_, j) => traningDataset[j] * gradients[i]));

            const currentLearningRate = this.scheduler.getLearningRate(epochs);
            this.optimizer.setLearningRate(currentLearningRate);
            this.optimizer.updateParameters(layer.weights, weightGradients);
            this.optimizer.updateBias(layer.biases, gradients);
            /*******************************/
        }
    }

}