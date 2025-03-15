import {IdentityActivationFunction} from "../../src/domain/entities/activation-functions/IdentityActivationFunction";
import {Perceptron} from "../../src/domain/entities/Perceptron";
import {PerceptronTrainer} from "../../src/domain/entities/training/PerceptronTrainer";
import {BatchGradientDescent} from "../../src/domain/entities/BatchGradientDescent";
import {MeanSquareErrorLossFunction} from "../../src/domain/entities/training/MeanSquareErrorLossFunction";

describe("Trainer Tests", (): void => {
    const errorMargin: number = 0.001;
    const learningRate: number = 0.001;

    it("Should train with adaptativeLearningRate and from -50 to 50 as input with a linear activation function", () => {
        let weights: number[] = [0];
        let bias: number = 0;
        const activationFunction = new IdentityActivationFunction();
        const lossFunction = new MeanSquareErrorLossFunction();
        const trainingDataset = Array(2500).fill(0).map(() =>
            (Math.random() - 0.5) * 100
        );
        const targetFunction: Function = (input:number) => 2*input;

        const perceptron = new Perceptron(
            weights,
            bias,
            activationFunction,
        );

        let trainer = new PerceptronTrainer(new BatchGradientDescent());
        trainer.train(learningRate, errorMargin, perceptron, lossFunction, trainingDataset, targetFunction);
    })
})