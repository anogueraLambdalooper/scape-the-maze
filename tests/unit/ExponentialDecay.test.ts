import {ExponentialDecay} from "../../src/domain/entities/optimizers/ExponentialDecay";

describe("Exponential Decay", () => {
    it("Should update the learning rate correctly on 2 epochs", () => {
        let initialLearningRate = 0.1;
        let decay = 0.9;

        const exponentialDecay = new ExponentialDecay(initialLearningRate, decay);

        expect(exponentialDecay.getLearningRate(2)).toBeCloseTo(0.08);
    });

    it("Should update the learning rate correctly on 10 epochs", () => {
        let initialLearningRate = 0.1;
        let decay = 0.9;

        const exponentialDecay = new ExponentialDecay(initialLearningRate, decay);

        expect(exponentialDecay.getLearningRate(10)).toBeCloseTo(0.034);
    });

    it("Should update the learning rate correctly on 100 epochs", () => {
        let initialLearningRate = 0.1;
        let decay = 0.9;

        const exponentialDecay = new ExponentialDecay(initialLearningRate, decay);

        expect(exponentialDecay.getLearningRate(100)).toBeCloseTo(0.00026);
    });
});