import {Perceptron} from "../../src/domain/entities/Perceptron";
import {ActivationFunction} from "../../src/domain/interfaces/ActivationFunction";
import {Tanh} from "../../src/domain/entities/Tanh";
import {ReLU} from "../../src/domain/entities/ReLU";
import {Sigmoid} from "../../src/domain/entities/Sigmoid";

describe('PerceptronTest', () => {
    let mockPerceptron: Perceptron;
    let mockActivationFunction: ActivationFunction;

    it('Should throw Error when there are more inputs than weights', () => {
        let inputs = [0,1,4];
        let weights = [4,3];
        let bias = 1;
        mockActivationFunction = new Tanh();

        mockPerceptron = new Perceptron(weights, bias, mockActivationFunction);

        expect(() => mockPerceptron.forward(inputs)).toThrow("Missmatch between inputs and weights length");
    })

    it('Should throw Error when there are more weights than inputs', () => {
        let inputs = [0,1];
        let weights = [4,3,2];
        let bias = 1;
        mockActivationFunction = new Tanh();

        mockPerceptron = new Perceptron(weights, bias, mockActivationFunction);

        expect(() => mockPerceptron.forward(inputs)).toThrow("Missmatch between inputs and weights length");
    })

    it('Should output be close 0.9993 when using Tanh as AF', () => {
        let inputs = [0,1];
        let weights = [4,3];
        let bias = 1;
        mockActivationFunction = new Tanh();

        mockPerceptron = new Perceptron(weights, bias, mockActivationFunction);

        expect(mockPerceptron.forward(inputs)).toBeCloseTo(0.9993);
    })

    it('Should output be 4 when using ReLU as AF', () => {
        let inputs = [0,1];
        let weights = [4,3];
        let bias = 1;
        mockActivationFunction = new ReLU();

        mockPerceptron = new Perceptron(weights, bias, mockActivationFunction);

        expect(mockPerceptron.forward(inputs)).toBe(4);
    })

    it('Should output be close 0.9820 when using Sigmoid as AF', () => {
        let inputs = [0,1];
        let weights = [4,3];
        let bias = 1;
        mockActivationFunction = new Sigmoid();

        mockPerceptron = new Perceptron(weights, bias, mockActivationFunction);

        expect(mockPerceptron.forward(inputs)).toBeCloseTo(0.9820);
    })
})