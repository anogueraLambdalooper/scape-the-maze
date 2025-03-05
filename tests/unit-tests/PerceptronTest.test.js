const { Perceptron } = require("../../src/domain/entities/Perceptron.js");

const inputs = [0,1,2,3,4];
const weights = [4,3,2,1,0];
const bias = 1;

const g = () => {
    let output = 0;
    for(let i = 0; i < inputs.length; i++) {
        output += inputs[i] * weights[i];
    }
    output += bias;
    return output;
}

const perceptron = new Perceptron(inputs, weights, bias, g);

test('The output of the perceptron is 11', () => {
    expect(perceptron.getOutput()).toBe(11);
});

test('The output of the perceptron is not 21', () => {
    expect(perceptron.getOutput()).not.toBe(21);
})