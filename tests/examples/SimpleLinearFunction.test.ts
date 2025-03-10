import {Perceptron} from "../../src/domain/entities/Perceptron";
import {IdentityActivationFunction} from "../../src/domain/entities/activation-functions/IdentityActivationFunction";

describe("Linear Function", () => {

    //y=2x
    it("Should measure performance", () => {
        let weights = [0];
        let bias = 0;
        const mockActivationFunction = new IdentityActivationFunction();

        const perceptron = new Perceptron(
            weights,
            bias,
            mockActivationFunction,
            0.0001
        );

        while(true) {
            let input = [(Math.random() - 0.5) * 100] //Input space from -50 to 50
            let target = 2 * input[0] + 3
            const output = perceptron.forward(input);

            const error = target - output;

            //console.info("Error", Math.abs(error));

            if (Math.abs(error) < 0.01) {
                console.info("Error", Math.abs(error));
                console.info("WEIGHTS", perceptron.weights, perceptron.bias);
                console.info("Solved!");
                break;
            }

            perceptron.backward(input, target);
        }
    })
})