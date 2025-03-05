class MLP {
    constructor(inputLayer, hiddenLayers, outputLayer) {
        this.inputLayer = inputLayer;
        this.hiddenLayers = hiddenLayers;
        this.outputLayer = outputLayer;
        this.layers = [inputLayer, hiddenLayers, outputLayer];
    }

    forward(input) {
        output = 0;
        for(let i = 0; i < this.inputLayer.length; i++) {
            for(let i = 0; i < this.hiddenLayers.length; i++) {
                this.hiddenLayers.
            }
        }
    }
}

module.exports = { MLP };