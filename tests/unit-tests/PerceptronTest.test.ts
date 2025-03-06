import {Perceptron} from "../../src/domain/entities/Perceptron";
import {ActivationFunction} from "../../src/domain/interfaces/ActivationFunction";
import {Tanh} from "../../src/domain/entities/Tanh";

describe('PerceptronTest', () => {
    let mockPerceptron: Perceptron;
    let mockActivationFunction: ActivationFunction;

    it('Should output 21', () => {
        //Arrange
        let inputs = [0,1,2,3,4];
        let weights = [4,3,2,1,0];
        let bias = 1;
        mockActivationFunction = new Tanh();

        mockPerceptron = new Perceptron(weights, bias, mockActivationFunction);

        expect(mockPerceptron.forward(inputs)).toEqual(0.7615941559557649);
    })
})