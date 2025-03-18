import {IdentityActivationFunction} from "../../src/domain/entities/activation-functions/IdentityActivationFunction";
import {Perceptron} from "../../src/domain/entities/Perceptron";
import {PerceptronTrainer} from "../../src/domain/entities/training/PerceptronTrainer";
import {SGD} from "../../src/domain/entities/optimizers/SGD";
import {MeanSquareErrorLossFunction} from "../../src/domain/entities/training/MeanSquareErrorLossFunction";

describe("Trainer Tests", (): void => {
    const errorMargin: number = 0.001;
    const learningRate: number = 0.001;
    const decayRate: number = 0.01;

    it("Should train with adaptativeLearningRate and from -50 to 50 as input with a linear activation function", () => {
        let weights: number[] = [0];
        let bias: number = 0;
        const activationFunction = new IdentityActivationFunction();
        const lossFunction = new MeanSquareErrorLossFunction();
        const trainingDataset = Array(300).fill(0).map(() =>
            (Math.random() - 0.5) * 100
        );
        const targetFunction: Function = (input: number) => 2 * input;

        const optimizer = new SGD(learningRate, decayRate, 0.0001, 0.9);

        const perceptron = new Perceptron(
            weights,
            bias
        );

        let trainer = new PerceptronTrainer(optimizer);
        const finished = trainer.train(errorMargin, perceptron, lossFunction, trainingDataset, targetFunction, activationFunction);

        expect(finished).toBe(true);
    })
})