import { Perceptron } from "../../src/domain/entities/Perceptron";
import { ActivationFunction } from "../../src/domain/interfaces/ActivationFunction";
import { Tanh } from "../../src/domain/entities/activation-functions/Tanh";
import { ReLU } from "../../src/domain/entities/activation-functions/ReLU";
import { Sigmoid } from "../../src/domain/entities/activation-functions/Sigmoid";

describe("Perceptron", () => {
  let mockPerceptron: Perceptron;
  let mockActivationFunction: ActivationFunction;

  it("Should throw Error when there are more inputs than weights", () => {
    let inputs = [0, 1, 4];
    let weights = [4, 3];
    let bias = 1;
    let learningRate = 0.1;
    mockActivationFunction = new Tanh();

    mockPerceptron = new Perceptron(
      weights,
      bias,
      mockActivationFunction,
      learningRate
    );

    expect(() => mockPerceptron.forward(inputs)).toThrow(
      "Missmatch between inputs and weights length"
    );
  });

  it("Should throw Error when there are more weights than inputs", () => {
    let inputs = [0, 1];
    let weights = [4, 3, 2];
    let bias = 1;
    let learningRate = 0.1;
    mockActivationFunction = new Tanh();

    mockPerceptron = new Perceptron(
      weights,
      bias,
      mockActivationFunction,
      learningRate
    );

    expect(() => mockPerceptron.forward(inputs)).toThrow(
      "Missmatch between inputs and weights length"
    );
  });

  it("Should output be close 0.9993 when using Tanh as AF", () => {
    let inputs = [0, 1];
    let weights = [4, 3];
    let bias = 1;
    let learningRate = 0.1;
    mockActivationFunction = new Tanh();

    mockPerceptron = new Perceptron(
      weights,
      bias,
      mockActivationFunction,
      learningRate
    );

    expect(mockPerceptron.forward(inputs)).toBeCloseTo(0.9993);
  });

  it("Should output be 4 when using ReLU as AF", () => {
    let inputs = [0, 1];
    let weights = [4, 3];
    let bias = 1;
    let learningRate = 0.1;
    mockActivationFunction = new ReLU();

    mockPerceptron = new Perceptron(
      weights,
      bias,
      mockActivationFunction,
      learningRate
    );

    expect(mockPerceptron.forward(inputs)).toBe(4);
  });

  it("Should output be close 0.9820 when using Sigmoid as AF", () => {
    let inputs = [0, 1];
    let weights = [4, 3];
    let bias = 1;
    let learningRate = 0.1;
    mockActivationFunction = new Sigmoid();

    mockPerceptron = new Perceptron(
      weights,
      bias,
      mockActivationFunction,
      learningRate
    );

    expect(mockPerceptron.forward(inputs)).toBeCloseTo(0.982);
  });

  it("Should have updated the weights and the bias on the backward pass using Tahn AF", () => {
    let inputs = [2, 1];
    let weights = [4, 3];
    let oldWeights = [4, 3];
    let oldBias = 1;

    let bias = 1;
    let learningRate = 10;
    let target = 0.5; //Tanh target must be between -1 and 1
    mockActivationFunction = new Tanh();

    mockPerceptron = new Perceptron(
      weights,
      bias,
      mockActivationFunction,
      learningRate
    );

    mockPerceptron.backward(inputs, target);

    expect(mockPerceptron.weights[0]).not.toEqual(oldWeights[0]);
    expect(mockPerceptron.weights[1]).not.toEqual(oldWeights[1]);
    expect(mockPerceptron.bias).not.toEqual(oldBias);
  });

  it("Should have updated the weights and the bias on the backward pass using Sigmoid AF", () => {
    let inputs = [2, 1];
    let weights = [4, 3];
    let oldWeights = [4, 3];
    let oldBias = 1;

    let bias = 1;
    let learningRate = 10;
    let target = 0.5; //Sigmoid target must be between 0 and 1
    mockActivationFunction = new Sigmoid();

    mockPerceptron = new Perceptron(
        weights,
        bias,
        mockActivationFunction,
        learningRate
    );

    mockPerceptron.backward(inputs, target);

    expect(mockPerceptron.weights[0]).not.toEqual(oldWeights[0]);
    expect(mockPerceptron.weights[1]).not.toEqual(oldWeights[1]);
    expect(mockPerceptron.bias).not.toEqual(oldBias);
  });

  it("Should have updated the weights and the bias on the backward pass using ReLU AF", () => {
    let inputs = [2, 1];
    let weights = [4, 3];
    let oldWeights = [4, 3];
    let oldBias = 1;

    let bias = 1;
    let learningRate = 10;
    let target = 0.5; //ReLU target must not be negative
    mockActivationFunction = new ReLU();

    mockPerceptron = new Perceptron(
        weights,
        bias,
        mockActivationFunction,
        learningRate
    );

    mockPerceptron.backward(inputs, target);

    expect(mockPerceptron.weights[0]).not.toEqual(oldWeights[0]);
    expect(mockPerceptron.weights[1]).not.toEqual(oldWeights[1]);
    expect(mockPerceptron.bias).not.toEqual(oldBias);
  });

  it("Should have updated gradient on the backward", () => {
    let inputs = [2, 1];
    let weights = [4, 3];

    let bias = 1;
    let learningRate = 10;
    let target = 0.5;
    mockActivationFunction = new ReLU();

    mockPerceptron = new Perceptron(
        weights,
        bias,
        mockActivationFunction,
        learningRate
    );

    mockPerceptron.backward(inputs, target);

    expect(mockPerceptron.localGradient).not.toEqual(0);
  });
});
