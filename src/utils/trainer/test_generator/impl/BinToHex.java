package src.utils.trainer.test_generator.impl;

import src.Setup;
import src.utils.trainer.test_generator.TestGenerator;

public class BinToHex implements TestGenerator {

    @Override
    public Double[][][] generateTrainingData() {
        Integer DEMO_TRAINING_DATA_LENGTH = Setup.getDemoTrainingDataLength();
        Integer[] LAYER_SIZES = Setup.getLayerSizes();
        Double[][][] TRAINING_DATA = new Double[DEMO_TRAINING_DATA_LENGTH][2][LAYER_SIZES[0] > LAYER_SIZES[LAYER_SIZES.length - 1] ? LAYER_SIZES[0] : LAYER_SIZES[LAYER_SIZES.length - 1]];
        // Generate training data for the BinToHex conversion
        for (int i = 0; i < DEMO_TRAINING_DATA_LENGTH; i++) {
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
