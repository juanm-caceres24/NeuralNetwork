package v1.src.utils.trainer.test_generator.impl;

import v1.src.Setup;
import v1.src.utils.trainer.test_generator.TestGenerator;

public class BinToHex implements TestGenerator {

    @Override
    public double[][][] generateTrainingData() {
        int TEST_TRAINING_DATA_LENGTH = Setup.getTestTrainingDataLength();
        int[] LAYERS_SIZE = Setup.getLayersSize();
        double[][][] TRAINING_DATA = new double[TEST_TRAINING_DATA_LENGTH][2][LAYERS_SIZE[LAYERS_SIZE.length - 1]];
        // Generate training data for the BinToHex conversion
        for (int i = 0; i < TEST_TRAINING_DATA_LENGTH; i++) {
            // Generate random binary input
            double[] inputs = TRAINING_DATA[i][0];
            // initialize all positions as NaN to mark 'unused' slots
            for (int k = 0; k < inputs.length; k++) inputs[k] = Double.NaN;
            for (int j = 0; j < Setup.getLayersSize()[0]; j++) {
                inputs[j] = Math.random();
            }
            // Convert binary input to hexadecimal output
            double[] outputs = TRAINING_DATA[i][1];
            int decimalValue = 0;
            int inSize = Setup.getLayersSize()[0];
            // Convert binary-like inputs to decimal value
            for (int j = 0; j < inSize; j++) {
                int bit = (inputs[j] > 0.5) ? 1 : 0;
                if (bit == 1) {
                    decimalValue += 1 << j;
                }
            }
            int outSize = Setup.getLayersSize()[Setup.getLayersSize().length - 1];
            for (int j = 0; j < outSize; j++) {
                outputs[j] = (j == decimalValue) ? 1.0 : 0.0;
            }
        }
        return TRAINING_DATA;
    }
}
