import {Perceptron} from "../../src/domain/entities/Perceptron";
import {IdentityActivationFunction} from "../../src/domain/entities/activation-functions/IdentityActivationFunction";
import {Sigmoid} from "../../src/domain/entities/activation-functions/Sigmoid";
import {ReLU} from "../../src/domain/entities/activation-functions/ReLU";

describe("Linear Function", () => {

    //y=2x
    it("Should arrive to the solution of y=2x", () => {
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
            let target = 2 * input[0]
            const output = perceptron.forward(input);

            const error = target - output;

            if (Math.abs(error) < 0.01) {
                console.info("Error", Math.abs(error));
                console.info("WEIGHTS", perceptron.weights, perceptron.bias);
                console.info("Solved!");
                break;
            }

            perceptron.backward(input, target);
        }
    });

    it("Should arrive to the solution of y=2x + 3", () => {
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

            if (Math.abs(error) < 0.01) {
                console.info("Error", Math.abs(error));
                console.info("WEIGHTS", perceptron.weights, perceptron.bias);
                console.info("Solved!");
                break;
            }

            perceptron.backward(input, target);
        }
    });

    it("Should arrive to the solution of y=x * x + 2", () => {
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
            let target = input[0] * input[0] + 2
            const output = perceptron.forward(input);

            const error = target - output;

            if (Math.abs(error) < 0.01) {
                console.info("Error", Math.abs(error));
                console.info("WEIGHTS", perceptron.weights, perceptron.bias);
                console.info("Solved!");
                break;
            }

            perceptron.backward(input, target);
        }
    });

    it("Should arrive to the solution of y=x**2/(1 + x**2) with a Sigmoid", () => {
        let weights = [0];
        let bias = 0;
        const mockActivationFunction = new Sigmoid();

        const perceptron = new Perceptron(
            weights,
            bias,
            mockActivationFunction,
            0.1
        );

        while(true) {
            let input = [(Math.random() - 0.5) * 100] //Input space from -50 to 50
            let target = input[0] ** 2 / (1 + input[0] ** 2);
            const output = perceptron.forward(input);

            const error = target - output;
            console.log(Math.abs(error));

            if (Math.abs(error) < 0.01) {
                console.info("Error", Math.abs(error));
                console.info("WEIGHTS", perceptron.weights, perceptron.bias);
                console.info("Solved!");
                break;
            }

            perceptron.backward(input, target);
        }
    });

    it("Should arrive to the solution of max(0, 2x+3) with ReLU", () => {
        let weights = [0];
        let bias = 1;
        const mockActivationFunction = new ReLU();

        const perceptron = new Perceptron(
            weights,
            bias,
            mockActivationFunction,
            0.1
        );

        while(true) {
            let input = [(Math.random() - 0.5) * 100] //Input space from -50 to 50
            let target = Math.max(0, 2 * input[0] + 3);
            const output = perceptron.forward(input);

            const error = target - output;

            if (Math.abs(error) < 0.01) {
                console.info("Error", Math.abs(error));
                console.info("WEIGHTS", perceptron.weights, perceptron.bias);
                console.info("Solved!");
                break;
            }

            perceptron.backward(input, target);
        }
    });

    it("Should arrive to the solution of max(0, sin(x)) with ReLU", () => {
        let weights = [0];
        let bias = 1;
        const mockActivationFunction = new ReLU();

        const perceptron = new Perceptron(
            weights,
            bias,
            mockActivationFunction,
            0.1
        );

        while(true) {
            let input = [(Math.random() - 0.5) * 100] //Input space from -50 to 50
            let target = Math.max(0, Math.sin(input[0]));
            const output = perceptron.forward(input);

            const error = target - output;

            if (Math.abs(error) < 0.01) {
                console.info("Error", Math.abs(error));
                console.info("WEIGHTS", perceptron.weights, perceptron.bias);
                console.info("Solved!");
                break;
            }

            perceptron.backward(input, target);
        }
    });

    it("Should arrive to the solution of tanh(0.5 * x) with ReLU", () => {
        let weights = [0];
        let bias = 1;
        const mockActivationFunction = new ReLU();

        const perceptron = new Perceptron(
            weights,
            bias,
            mockActivationFunction,
            0.001
        );

        while(true) {
            let input = [(Math.random() - 0.5) * 100] //Input space from -50 to 50
            let target = Math.tanh(0.5 * input[0]);
            const output = perceptron.forward(input);

            const error = target - output;
            console.info("Error in curse", error);

            if (Math.abs(error) < 0.01) {
                console.info("Error", Math.abs(error));
                console.info("WEIGHTS", perceptron.weights, perceptron.bias);
                console.info("Solved!");
                break;
            }

            perceptron.backward(input, target);
        }
    });
})