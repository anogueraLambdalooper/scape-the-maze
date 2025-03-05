class MLP {
    constructor(inputLayer, hiddenLayers, outputLayer) {
        this.inputLayer = inputLayer;
        this.hiddenLayers = hiddenLayers;
        this.outputLayer = outputLayer;
        this.layers = [inputLayer, hiddenLayers, outputLayer];
    }


}

module.exports = { MLP };