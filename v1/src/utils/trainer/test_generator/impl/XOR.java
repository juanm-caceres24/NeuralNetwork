package v1.src.utils.trainer.test_generator.impl;

import v1.src.Setup;
import v1.src.utils.trainer.test_generator.TestGenerator;

public class XOR implements TestGenerator {

    @Override
    public double[][][] generateTrainingData() {
        int TEST_TRAINING_DATA_LENGTH = Setup.getTestTrainingDataLength();
        int[] LAYERS_SIZE = Setup.getLayersSize();
        double[][][] TRAINING_DATA = new double[TEST_TRAINING_DATA_LENGTH][2][LAYERS_SIZE[0] > LAYERS_SIZE[LAYERS_SIZE.length - 1] ? LAYERS_SIZE[0] : LAYERS_SIZE[LAYERS_SIZE.length - 1]];
        // Generate random n-inputs between 0.0 and 1.0
        for (int i = 0; i < TEST_TRAINING_DATA_LENGTH; i++) {
            double[] inputs = TRAINING_DATA[i][0];
            for (int j = 0; j < inputs.length; j++) {
                inputs[j] = Math.random();
            }
        }
        // Generate training outputs as XOR of the inputs (set all the outputs to 0.0 or 1.0) (use modular code to n-inputs and m-outputs)
        for (int i = 0; i < TEST_TRAINING_DATA_LENGTH; i++) {
            double[] inputs = TRAINING_DATA[i][0];
            double[] outputs = TRAINING_DATA[i][1];
            // XOR logic: output is 1.0 if an odd number of inputs are > 0.5, else 0.0
            int count = 0;
            for (double input : inputs) {
                if (input > 0.5) count++;
            }
            for (int j = 0; j < outputs.length; j++) {
                if (j % 2 == 1) {
                    outputs[j] = (count % 2 == 0) ? 1.0 : 0.0;
                } else {
                    outputs[j] = (count % 2 == 1) ? 1.0 : 0.0;
                }
            }
        }
        return TRAINING_DATA;
    }
}
