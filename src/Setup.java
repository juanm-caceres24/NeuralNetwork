package src;

public class Setup {

    /*
     * ATTRIBUTES
     */

    private static Double[] INPUT_VALUES = {
        0.37, // N0
        1.00, // N1
        0.92, // N2
    };
    private static Double[][] BIASES = {
    //    N0   N1   N2   N3
        { 0.0, 0.0, 0.0 },      // i_L
        { 0.2, 0.3, 0.1, 0.4 }, // h_L0
        { 0.1, 0.2, 0.3, 0.2 }, // h_L1
        { 0.3, 0.1, 0.2, 0.3 }, // h_L2
        { 0.2, 0.2, 0.2, 0.2 }, // h_L3
        { 0.0, 0.0 }            // o_L
    };
    private static Double[][][] WEIGHTS = {
    //      W0_0  W0_1  W0_2  W0_3     W1_0  W1_1  W1_2  W1_3     W2_0  W2_1  W2_2  W2_3     W3_0  W3_1  W3_2  W3_3
        { { 0.10, 0.20, 0.30, 0.25 }, { 0.40, 0.10, 0.20, 0.15 }, { 0.25, 0.30, 0.35, 0.10 }, },                            // i_L
        { { 0.20, 0.10, 0.30, 0.25 }, { 0.15, 0.35, 0.25, 0.10 }, { 0.10, 0.20, 0.15, 0.30 }, { 0.25, 0.40, 0.20, 0.20 } }, // h_L0
        { { 0.30, 0.25, 0.15, 0.10 }, { 0.10, 0.35, 0.20, 0.25 }, { 0.25, 0.10, 0.30, 0.15 }, { 0.20, 0.25, 0.10, 0.35 } }, // h_L1
        { { 0.20, 0.15, 0.25, 0.30 }, { 0.25, 0.30, 0.10, 0.20 }, { 0.35, 0.25, 0.20, 0.10 }, { 0.10, 0.20, 0.30, 0.25 } }, // h_L2
        { { 0.30, 0.25 },             { 0.20, 0.30 },             { 0.25, 0.15 },             { 0.35, 0.10 } }              // o_L
    };
    private static Integer[] ACTIVATION_FUNCTIONS = {
        0, // i_L
        1, // h_Ls
        1  // o_L
    };
    private static Double LEARNING_RATE = 0.01;
    private static Integer NUMBER_OF_INPUTS = BIASES[0].length;
    private static Integer NUMBER_OF_OUTPUTS = BIASES[BIASES.length - 1].length;
    private static Integer NUMBER_OF_HIDDEN_LAYERS = BIASES.length - 2;
    private static Integer NEURONS_PER_HIDDEN_LAYER = BIASES[1].length;
    private static Integer TOTAL_NUMBER_OF_LAYERS = BIASES.length;
    
    /*
     * GETTERS AND SETTERS
     */

    public static Double[] getInputValues() { return INPUT_VALUES; }
    
    public static Double[][] getBiases() { return BIASES; }

    public static Double[][][] getWeights() { return WEIGHTS; }

    public static Integer[] getActivationFunctions() { return ACTIVATION_FUNCTIONS; }

    public static Double getLearningRate() { return LEARNING_RATE; }

    public static Integer getNumberOfInputs() { return NUMBER_OF_INPUTS; }

    public static Integer getNumberOfOutputs() { return NUMBER_OF_OUTPUTS; }

    public static Integer getNumberOfHiddenLayers() { return NUMBER_OF_HIDDEN_LAYERS; }

    public static Integer getNeuronsPerHiddenLayer() { return NEURONS_PER_HIDDEN_LAYER; }

    public static Integer getTotalNumberOfLayers() { return TOTAL_NUMBER_OF_LAYERS; }
}
