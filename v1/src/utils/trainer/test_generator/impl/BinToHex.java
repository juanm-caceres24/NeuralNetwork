package v1.src.utils.trainer.test_generator.impl;

import v1.src.Setup;
import v1.src.utils.trainer.test_generator.TestGenerator;

public class BinToHex implements TestGenerator {

    @Override
    public Double[][][] generateTrainingData() {
        Integer TEST_TRAINING_DATA_LENGTH = Setup.getTestTrainingDataLength();
        Integer[] LAYERS_SIZE = Setup.getLayersSize();
        Double[][][] TRAINING_DATA = new Double[TEST_TRAINING_DATA_LENGTH][2][LAYERS_SIZE[0] > LAYERS_SIZE[LAYERS_SIZE.length - 1] ? LAYERS_SIZE[0] : LAYERS_SIZE[LAYERS_SIZE.length - 1]];
        // Generate training data for the BinToHex conversion
        for (int i = 0; i < TEST_TRAINING_DATA_LENGTH; i++) {
            // Generate random 4-bit binary input
            Double[] inputs = TRAINING_DATA[i][0];
            for (int j = 0; j < 4; j++) {
                inputs[j] = Math.random() < 0.5 ? 0.0 : 1.0;
            }
            // Convert binary input to hexadecimal output
            Double[] outputs = TRAINING_DATA[i][1];
            int decimalValue = 0;
            for (int j = 0; j < 4; j++) {
                decimalValue += inputs[j].intValue() * (1 << (3 - j));
            }
            for (int j = 0; j < 16; j++) {
                outputs[j] = (j == decimalValue) ? 1.0 : 0.0;
            }
        }
        return TRAINING_DATA;
    }
}
