import {Perceptron} from "../../src/domain/entities/Perceptron";
import expect from "expect";

describe("Perceptron Perfomance", () => {

    it("Should measure performance", () => {
        let weights = Array(1000).fill(Math.random() * 100);
        let bias = Math.random() * 100;

        const perceptron = new Perceptron(
            weights,
            bias
        );

        for(let i = 0; i < 1000000; i++) {
            let inputs = Array(1000).fill(Math.random() * 100);
            perceptron.forward(inputs);
        }

        expect(true).toBe(true);
    })
})