const { Perceptron } = require("../../src/domain/entities/Perceptron.js");


const perceptron = new Perceptron([0,1,2,3,4], [4,3,2,1,0], 1);
test('The output of the perceptron is 21', () => {
    expect(perceptron.ActivationFunction()).toBe(21);
});