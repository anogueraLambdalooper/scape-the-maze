class Perceptron {
    constructor(inputs, weights, bias) {
        this.inputs = inputs;
        this.weights = weights;
        this.bias = bias;
    }

    GetOutput() {
        return this.ActivationFunction();
    }

    ActivationFunction() {
        let output = 0;
        for(let i = 0; i < this.inputs.length; i++) {
            output += this.inputs[i] + this.weights[i];
        }
        output += this.bias;
        return output;
    }
}

module.exports = { Perceptron };