import {Perceptron} from "../../src/domain/entities/Perceptron";
import {IdentityActivationFunction} from "../../src/domain/entities/activation-functions/IdentityActivationFunction";
import {CanvasService} from "../../src/application/services/CanvasService";

describe("Linear Function", () => {
    const errorMargin: number = 0.001;
    const learningRate: number = 0.001;

    //y=2x
    it("Should arrive to the solution of y=2x", () => {
        let weights = [0];
        let bias = 0;
        const mockActivationFunction = new IdentityActivationFunction();

        const perceptron = new Perceptron(
            weights,
            bias,
            mockActivationFunction
        );

        let epochs = 0;
        let errorsHistory = [];

        while(true) {
            let input = [(Math.random() - 0.5) * 100] //Input space from -50 to 50
            let target = 2 * input[0] // w=2 & b=0
            const output = perceptron.forward(input);

            const error = target - output;
            errorsHistory.push(error);
            epochs++;

            if (Math.abs(error) < errorMargin) {
                console.info("Error", Math.abs(error));
                console.info("WEIGHTS", perceptron.weights, perceptron.bias);
                console.info("Solved!");
                break;
            }

            perceptron.backward(input, target, learningRate);
        }

        const canvas = new CanvasService();
        canvas.printCanvas(errorsHistory, epochs, "Should arrive to the solution of y=2x");
    });

    it("Should arrive to the solution of y=2x + 3", () => {
        function target(input: number) : number {
            return 2 * input + 3;
        }

        let weights = [0];
        let bias = 0;
        const mockActivationFunction = new IdentityActivationFunction();

        const perceptron = new Perceptron(
            weights,
            bias,
            mockActivationFunction
        );

        let epochs = 0;
        let errorsHistory = [];

        while(true) {
            let input = [(Math.random() - 0.5) * 100] //Input space from -50 to 50
            let target_value = target(input[0]); // ha de conseguir un w = 2 y un b = 3
            const output = perceptron.forward(input);

            const error = target_value - output;
            errorsHistory.push(error);
            epochs++;

            if (Math.abs(error) < errorMargin) {
                console.info("Error", Math.abs(error));
                console.info("WEIGHTS", perceptron.weights, perceptron.bias);
                console.info("Solved!");
                break;
            }

            perceptron.backward(input, target_value, learningRate);
        }

        const canvas = new CanvasService();
        canvas.printCanvas(errorsHistory, epochs, "Should arrive to the solution of y=2x + 3");
    });

    it("Should arrive to the solution of y=2x in bigger range", () => {
        let weights = [0];
        let bias = 0;
        const mockActivationFunction = new IdentityActivationFunction();

        const perceptron = new Perceptron(
            weights,
            bias,
            mockActivationFunction
        );

        let epochs = 0;
        let errorsHistory = [];

        while(true) {
            let input = [(Math.random() - 0.5) * 100] //Input space from -500 to 500
            let target = 2 * input[0] // w=2 & b=0
            const output = perceptron.forward(input);

            const error = output - target;

            errorsHistory.push(error);
            epochs++;

            if (Math.abs(error) < errorMargin) {

                console.info("Error", Math.abs(error));
                console.info("WEIGHTS", perceptron.weights, perceptron.bias);
                console.info("Solved!");
                break;
            }

            perceptron.backward(input, target, 0.00001);
        }

        const canvas = new CanvasService();
        canvas.printCanvas(errorsHistory, epochs, "Should arrive to the solution of y=2x in bigger range");
    });

    it("Should arrive to the solution of y=2x+3 in bigger range", () => {
        let weights = [0];
        let bias = 0;
        const mockActivationFunction = new IdentityActivationFunction();

        const perceptron = new Perceptron(
            weights,
            bias,
            mockActivationFunction
        );

        let epochs = 0;
        let errorsHistory = [];

        while(true) {
            let input = [(Math.random() - 0.5) * 100] //Input space from -500 to 500
            let target = 2 * input[0] + 3 // w=2 & b=3
            const output = perceptron.forward(input);

            const error = output - target;
            errorsHistory.push(error);
            epochs++;

            if (Math.abs(error) < errorMargin) {

                console.info("Error", Math.abs(error));
                console.info("WEIGHTS", perceptron.weights, perceptron.bias);
                console.info("Solved!");
                break;
            }

            perceptron.backward(input, target, 0.00001);
        }

        const canvas = new CanvasService();
        canvas.printCanvas(errorsHistory, epochs, "Should arrive to the solution of y=2x+3 in bigger range");
    });

    it("Should arrive to the solution of y=2x in a even bigger range", () => {
        let weights = [0];
        let bias = 0;
        const mockActivationFunction = new IdentityActivationFunction();

        const perceptron = new Perceptron(
            weights,
            bias,
            mockActivationFunction
        );

        let epochs = 0;
        let errorsHistory = [];

        while(true) {
            let input = [(Math.random() - 0.5) * 1000] //Input space from -5000 to 5000
            let target = 2 * input[0] // w=2 & b=0
            const output = perceptron.forward(input);

            const error = output - target;

            errorsHistory.push(error);
            epochs++;

            if (Math.abs(error) < errorMargin) {

                console.info("Error", Math.abs(error));
                console.info("WEIGHTS", perceptron.weights, perceptron.bias);
                console.info("Solved!");
                break;
            }

            perceptron.backward(input, target, 0.0000001);
        }

        const canvas = new CanvasService();
        canvas.printCanvas(errorsHistory, epochs, "Should arrive to the solution of y=2x in a even bigger range");
    });

    it("Should arrive to the solution of y=2x+3 in an even bigger range", () => {
        let weights = [0];
        let bias = 0;
        const mockActivationFunction = new IdentityActivationFunction();

        const perceptron = new Perceptron(
            weights,
            bias,
            mockActivationFunction
        );

        while(true) {
            let input = [(Math.random() - 0.5) * 1000] //Input space from -5000 to 5000
            let target = 2 * input[0] + 3 // w=2 & b=3
            const output = perceptron.forward(input);

            const error = output - target;

            if (Math.abs(error) < errorMargin) {

                console.info("Error", Math.abs(error));
                console.info("WEIGHTS", perceptron.weights, perceptron.bias);
                console.info("Solved!");
                break;
            }

            perceptron.backward(input, target, 0.0000001);
        }
    });

    it("Should arrive to the solution of y=2x in a even muuuuuuch bigger range", () => {
        let weights = [0];
        let bias = 0;
        const mockActivationFunction = new IdentityActivationFunction();

        const perceptron = new Perceptron(
            weights,
            bias,
            mockActivationFunction
        );

        while(true) {
            let input = [(Math.random() - 0.5) * 10000] //Input space from -50000 to 50000
            let target = 2 * input[0] // w=2 & b=0
            const output = perceptron.forward(input);

            const error = output - target;

            if (Math.abs(error) < errorMargin) {

                console.info("Error", Math.abs(error));
                console.info("WEIGHTS", perceptron.weights, perceptron.bias);
                console.info("Solved!");
                break;
            }

            perceptron.backward(input, target, 0.000000001);
        }
    });

    it("Should arrive to the solution of y=2x+3 in an even muuuuuuch bigger range", () => {
        let weights = [0];
        let bias = 0;
        const mockActivationFunction = new IdentityActivationFunction();

        const perceptron = new Perceptron(
            weights,
            bias,
            mockActivationFunction
        );

        while(true) {
            let input = [(Math.random() - 0.5) * 10000] //Input space from -50000 to 50000
            let target = 2 * input[0] + 3 // w=2 & b=3
            const output = perceptron.forward(input);

            const error = output - target;

            if (Math.abs(error) < errorMargin) {

                console.info("Error", Math.abs(error));
                console.info("WEIGHTS", perceptron.weights, perceptron.bias);
                console.info("Solved!");
                break;
            }

            perceptron.backward(input, target, 0.000000001);
        }
    });
})