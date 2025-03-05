class LinearOperation {
    constructor(in_features, out_features) {
        let length = out_features + in_features;
        this.weights = [];
        this.bias = [];

        for(let i = 0; i < length; i++) {
            this.weights.push(Math.random() * 0.01);
        }
        
        for(let i = 0; i < out_features; i++) {
            this.bias.push(Math.random() * 0.01);
        }
    }

    forward(input) {
        
    }
}

module.exports = { Linear: LinearOperation };