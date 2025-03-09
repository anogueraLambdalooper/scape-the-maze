import {Linear} from "../../src/domain/entities/layers/Linear";
import {Layer} from "../../src/domain/interfaces/Layer";
import {ActivationFunction} from "../../src/domain/interfaces/ActivationFunction";
import {Tanh} from "../../src/domain/entities/activation-functions/Tanh";

describe("Linear Layer", (): void => {
    let mockLayer: Layer;
    let mockActivationFunction: ActivationFunction;

    it("Should throw Error when there are more inputs than inputs in perceptrons", () => {
        let input = [0.5, -0.3, 4];
        let learningRate = 1;
        const inFeatures = 2;
        const outFeatures = 3;
        mockActivationFunction = new Tanh();
        mockLayer = new Linear(inFeatures, outFeatures, mockActivationFunction, learningRate);

        expect(() => mockLayer.forwardPass(input)).toThrow(
            "Missmatch between inputs and weights length"
        );
    })

    it("Should throw Error when there are less targets than perceptrons", () => {
        let targets = [1, 1];
        let learningRate = 1;
        const inFeatures = 2;
        const outFeatures = 3;
        mockActivationFunction = new Tanh();
        mockLayer = new Linear(inFeatures, outFeatures, mockActivationFunction, learningRate);

        expect(() => mockLayer.backwardPass(targets)).toThrow(
            "Missmatch between amount of targets and perceptrons"
        );
    })

    it("Should output not be undefined when doing forwardPass", () => {
        let input = [0.5, -0.3];
        let learningRate = 1;
        const inFeatures = 2;
        const outFeatures = 3;
        mockActivationFunction = new Tanh();
        mockLayer = new Linear(inFeatures, outFeatures, mockActivationFunction, learningRate);

        expect(mockLayer.forwardPass(input)).not.toBe(undefined);
    })

    it("Should a result with size equals to the output size when doing forwardPass", () => {
        let input = [0.5, -0.3];
        let learningRate = 1;
        const inFeatures = 2;
        const outFeatures = 3;
        mockActivationFunction = new Tanh();
        mockLayer = new Linear(inFeatures, outFeatures, mockActivationFunction, learningRate);

        expect(mockLayer.forwardPass(input).length).toBe(outFeatures);
    })

    it("Should output not be undefined when doing backwardPass", () => {
        let input = [34, 3];
        let targets = [1, -1, 0];
        let learningRate = 1;
        const inFeatures = 2;
        const outFeatures = 3;
        mockActivationFunction = new Tanh();
        mockLayer = new Linear(inFeatures, outFeatures, mockActivationFunction, learningRate);

        mockLayer.forwardPass(input);

        expect(mockLayer.backwardPass(targets)).not.toBe(undefined);
    })

    it("Should a result with size equals to the input size when doing backwardPass", () => {
        let input = [3, -2];
        let targets = [1, -1, 0];
        let learningRate = 1;
        const inFeatures = 2;
        const outFeatures = 3;
        mockActivationFunction = new Tanh();
        mockLayer = new Linear(inFeatures, outFeatures, mockActivationFunction, learningRate);

        mockLayer.forwardPass(input);

        expect(mockLayer.backwardPass(targets).length).toBe(inFeatures);
    })
});