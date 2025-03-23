import {IdentityActivationFunction} from "../../src/domain/entities/activation-functions/IdentityActivationFunction";
import {Perceptron} from "../../src/domain/entities/Perceptron";
import {PerceptronTrainer} from "../../src/domain/entities/training/PerceptronTrainer";
import {SGD} from "../../src/domain/entities/optimizers/SGD";
import {MeanSquareErrorLossFunction} from "../../src/domain/entities/training/MeanSquareErrorLossFunction";
import {ExponentialDecay} from "../../src/domain/entities/optimizers/ExponentialDecay";

describe("Trainer Tests", (): void => {
    const errorMargin: number = 0.001;
    const learningRate: number = 0.001;

    it("Should train with adaptativeLearningRate and from -50 to 50 as input with a linear activation function", () => {
        let weights: number[] = [0];
        let bias: number = 0;
        const activationFunction = new IdentityActivationFunction();
        const lossFunction = new MeanSquareErrorLossFunction();
        const trainingDataset = Array(300).fill(0).map(() =>
            (Math.random() - 0.5) * 100
        );
        const targetFunction: Function = (input: number) => 2 * input;
        let decay = 0.9;

        const optimizer = new SGD(learningRate);
        const scheduler = new ExponentialDecay(learningRate, decay)

        const perceptron = new Perceptron(
            weights,
            bias
        );

        let trainer = new PerceptronTrainer(optimizer, scheduler);
        const finished = trainer.train(errorMargin, perceptron, lossFunction, trainingDataset, targetFunction, activationFunction);

        expect(finished).toBe(true);
    })

    it("Should train with adaptativeLearningRate and from -500 to 500 as input with a linear activation function", () => {
        let weights: number[] = [0];
        let bias: number = 0;
        const activationFunction = new IdentityActivationFunction();
        const lossFunction = new MeanSquareErrorLossFunction();
        const trainingDataset = Array(300).fill(0).map(() =>
            (Math.random() - 0.5) * 1000
        );
        const targetFunction: Function = (input: number) => 2 * input;
        let decay = 0.9;

        const optimizer = new SGD(learningRate);
        const scheduler = new ExponentialDecay(learningRate, decay)

        const perceptron = new Perceptron(
            weights,
            bias
        );

        let trainer = new PerceptronTrainer(optimizer, scheduler);
        const finished = trainer.train(errorMargin, perceptron, lossFunction, trainingDataset, targetFunction, activationFunction);

        expect(finished).toBe(true);
    })
})