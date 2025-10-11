package src;

import src.utils.MSELoss;
import src.utils.Trainer;

public class Main {

    /*
     * ATTRIBUTES
     */

    private static Controller controller;
    private static Trainer trainer;

    /*
     * MAIN METHOD
     */
    public static void main(String args[]) {
        controller = new Controller();
        trainer = new Trainer(
                controller.getNetwork(),
                new MSELoss()
        );
        Double[][] X = new Double[][] {
            new Double[] {0.0, 0.0, 0.0},
            new Double[] {0.0, 0.0, 1.0},
            new Double[] {0.0, 1.0, 0.0},
            new Double[] {0.0, 1.0, 1.0},
            new Double[] {1.0, 0.0, 0.0},
            new Double[] {1.0, 0.0, 1.0},
            new Double[] {1.0, 1.0, 0.0},
            new Double[] {1.0, 1.0, 1.0}
        };
        Double[][] Y = new Double[][] {
            new Double[] {0.0, 0.0},
            new Double[] {1.0, 1.0},
            new Double[] {1.0, 1.0},
            new Double[] {0.0, 0.0},
            new Double[] {1.0, 1.0},
            new Double[] {0.0, 0.0},
            new Double[] {0.0, 0.0},
            new Double[] {1.0, 1.0}
        };
        trainer.train(X, Y);
        controller.predict();
        controller.getUserInterface().showNetwork();
        controller.getUserInterface().showInputs();
        controller.getUserInterface().showOutputs();
    }
}
