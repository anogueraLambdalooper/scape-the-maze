import { SGD } from "../../src/domain/entities/optimizers/SGD";
import { Perceptron } from "../../src/domain/entities/Perceptron";
import { MeanSquareErrorLossFunction } from "../../src/domain/entities/training/MeanSquareErrorLossFunction";

describe("SGD Optimizer with Perceptron", () => {
    it("should update perceptron weights and bias correctly", () => {
        const perceptron = new Perceptron([0.5, -0.5], 0.0);
        const sgd = new SGD(0.1);
        const lossFunction = new MeanSquareErrorLossFunction();

        const input = [1.0, 2.0];
        const target = 1.0;

        const output = perceptron.forward(input);
        const gradient = lossFunction.derivative(output, target);

        const gradients = input.map(x => x * gradient);
        gradients.push(gradient); // Adding bias gradient

        const parameters = [...perceptron.weights, perceptron.bias];
        sgd.updateParameters(parameters, gradients);

        perceptron.weights = parameters.slice(0, -1);
        perceptron.bias = parameters[parameters.length - 1];

        expect(perceptron.weights).toEqual([0.4, -0.7]);
        expect(perceptron.bias).toBe(-0.1);
    });
});