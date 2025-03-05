class Perceptron {
    constructor(weights, bias, activationFunction) {
        this.weights = weights;
        this.bias = bias;
        this.activationFunction = activationFunction;
    }

    getOutput() {
        return this.activationFunction();
    }
}

module.exports = { Perceptron };