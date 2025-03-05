const { MLP } = require("../../src/domain/entities/MLP.js");
const { Perceptron } = require("../../src/domain/entities/Perceptron.js");

const inputLayer = [new Perceptron()];
const hiddenLayers = [new Perceptron(), new Perceptron()];
const outputLayer = [new Perceptron()];

const mlp = new MLP(inputLayer, hiddenLayers, outputLayer);

test("The MLP should have 3 layers", () => {
    expect(mlp.layers.length).toBe(3);
})

test("The MLP should have 1 input layer", () =>  {
    expect(mlp.inputLayer.length).toBe(1);
})

test("The MLP should have 2 hidden layers", () => {
    expect(mlp.hiddenLayers.length).toBe(2);
})

test("The MLP should have 1 output layer", () =>  {
    expect(mlp.outputLayer.length).toBe(1);
})