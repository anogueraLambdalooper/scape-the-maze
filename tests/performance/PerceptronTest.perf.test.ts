import {Perceptron} from "../../src/domain/entities/Perceptron";
import {IdentityActivationFunction} from "../../src/domain/entities/activation-functions/IdentityActivationFunction";
import expect from "expect";

describe("Perceptron Perfomance", () => {

    it("Should measure performance", () => {
        let weights = Array(1000).fill(Math.random() * 100);
        let bias = Math.random() * 100;
        const mockActivationFunction = new IdentityActivationFunction();

        const perceptron = new Perceptron(
            weights,
            bias,
            mockActivationFunction,
            0.001
        );

        for(let i = 0; i < 1000000; i++) {
            let inputs = Array(1000).fill(Math.random() * 100);
            perceptron.forward(inputs);
        }

        expect(true).toBe(true);
    })
})