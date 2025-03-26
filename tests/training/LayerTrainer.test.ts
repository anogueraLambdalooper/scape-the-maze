import {IdentityActivationFunction} from "../../src/domain/entities/activation-functions/IdentityActivationFunction";
import {LayerMeanSquareErrorLossFunction} from "../../src/domain/entities/training/LayerMeanSquareErrorLossFunction";
import {LinearLayer} from "../../src/domain/entities/layers/LinearLayer";
import {LayerTrainer} from "../../src/domain/entities/training/LayerTrainer";
import {ExponentialDecay} from "../../src/domain/entities/optimizers/ExponentialDecay";
import {LayerSGD} from "../../src/domain/entities/optimizers/LayerSGD";

describe("Layer Trainer Tests", (): void => {
    const errorMargin: number = 0.001;
    const learningRate: number = 0.0001;

    it("Should train with 2x and from -50 to 50 as input with a linear activation function", () => {
        let weights: number[][] = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]];
        let biases: number[] = [0, 0, 0];
        const activationFunction = new IdentityActivationFunction();
        const lossFunction = new LayerMeanSquareErrorLossFunction();
        const trainingDataset = Array(15).fill(0).map(() =>
            (Math.random() - 0.5) * 100
        );
        const targetFunction: Function = (input: number) => 2 * input;
        let decay = 0.9;

        const optimizer = new LayerSGD(learningRate);
        const scheduler = new ExponentialDecay(learningRate, decay);

        const layer = new LinearLayer(
            weights,
            biases
        );

        let trainer = new LayerTrainer(optimizer, scheduler);
        const finished = trainer.train(errorMargin, layer, lossFunction, trainingDataset, targetFunction, activationFunction);

        expect(finished).toBe(true);
    });
});