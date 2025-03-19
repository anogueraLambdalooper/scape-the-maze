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

  it("Output should be 190", () => {
    let inputs = [-23, -5];
    let weights = [-8, -2];
    let bias = -4;

    let perceptron = new Perceptron(
        weights,
        bias
    );

    expect(perceptron.forward(inputs)).toBe(190);
  })

  it("Output should be 915", () => {
    let inputs = [-34, 5];
    let weights = [-14, 29];
    let bias = 294;

    let perceptron = new Perceptron(
        weights,
        bias
    );

    expect(perceptron.forward(inputs)).toBe(915);
  })

  it("Backward should return value 0", () => {
    let inputs = [-23, -5];
    let weights = [-8, -2];
    let bias = -4;

    let perceptron = new Perceptron(
        weights,
        bias
    );

    let output = perceptron.forward(inputs);
    let target = 190;

    expect(perceptron.backward(target, output)).toBe(0);
  })

  it("Backward should return value -190", () => {
    let inputs = [-23, -5];
    let weights = [-8, -2];
    let bias = -4;

    let perceptron = new Perceptron(
        weights,
        bias
    );

    let output = perceptron.forward(inputs);
    let target = 2 * output;

    expect(perceptron.backward(target, output)).toBe(190);
  })

  it("Backward should return value 450.5", () => {
    let inputs = [-4, -17];
    let weights = [2, 55];
    let bias = 42;

    let perceptron = new Perceptron(
        weights,
        bias
    );

    let output = perceptron.forward(inputs);
    let target = output/2;

    expect(perceptron.backward(target, output)).toBe(450.5);
  })
});
