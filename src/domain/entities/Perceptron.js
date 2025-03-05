class Perceptron {
    constructor(inputs, weights, bias, activationFunction) {
        this.inputs = inputs;
        this.weights = weights;
        this.bias = bias;
        this.activationFunction = activationFunction;
    }

    getOutput() {
        return this.activationFunction();
    }
}

module.exports = { Perceptron };