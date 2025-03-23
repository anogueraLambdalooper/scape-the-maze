import {SGD} from "../../src/domain/entities/optimizers/SGD";

describe("Stocastic Gradient Descent", () => {
    it("Should initialize with the correct learning rate", () => {
        const sgd = new SGD(0.01);
        expect(sgd.getLearningRate()).toBe(0.01);
    });

    it("Should update parameters correctly", () => {
        const sgd = new SGD(0.1);
        const parameters = [0.5, 1.5, -0.5];
        const gradients = [0.1, -0.2, 0.3];
        sgd.updateParameters(parameters, 1, gradients);
        expect(parameters).toEqual([0.49, 1.52, -0.53]);
    });

    it("Should set a new learning rate", () => {
        const sgd = new SGD(0.01);
        sgd.setLearningRate(0.05);
        expect(sgd.getLearningRate()).toBe(0.05);
    });
});