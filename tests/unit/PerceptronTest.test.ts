import { Perceptron } from "../../src/domain/entities/Perceptron";

describe("Perceptron", () => {

  it("Should throw Error when there are more inputs than weights", () => {
    let inputs = [0, 1, 4];
    let weights = [4, 3];
    let bias = 1;

    const perceptron = new Perceptron(
      weights,
      bias
    );

    expect(() => perceptron.forward(inputs)).toThrow(
      "Missmatch between inputs and weights length"
    );
  });

  it("Should throw Error when there are more weights than inputs", () => {
    let inputs = [0, 1];
    let weights = [4, 3, 2];
    let bias = 1;

    let perceptron = new Perceptron(
      weights,
      bias
    );

    expect(() => perceptron.forward(inputs)).toThrow(
      "Missmatch between inputs and weights length"
    );
  });

  it("Output should be 4", () => {
    let inputs = [0, 1];
    let weights = [4, 3];
    let bias = 1;

    let perceptron = new Perceptron(
      weights,
      bias
    );

    expect(perceptron.forward(inputs)).toBe(4);
  });

  it("Output should be 15", () => {
    let inputs = [2, 2];
    let weights = [4, 3];
    let bias = 1;

    let perceptron = new Perceptron(
      weights,
      bias
    );

    expect(perceptron.forward(inputs)).toBe(15);
  });

  it("Output should be 901", () => {
    let inputs = [-4, -17];
    let weights = [2, 55];
    let bias = 42;

    let perceptron = new Perceptron(
      weights,
      bias
    );

    expect(perceptron.forward(inputs)).toBe(-901);
  });

});
