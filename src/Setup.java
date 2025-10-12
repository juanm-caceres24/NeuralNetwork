package src;

public class Setup {

    /*
     * ATTRIBUTES
     */

    private static Double[] INPUT_VALUES = {
        1.00, // N0
        0.00,  // N1
        0.00  // N2
    };
    private static Double[][] BIASES = {
    //    N0    N1    N2    N3
        { 0.00, 0.00, 0.00 },       // i_L
        { 0.20, 0.30, 0.10, 0.40 }, // h_L0
        { 0.10, 0.20, 0.30, 0.20 }, // h_L1
        { 0.30, 0.10, 0.20, 0.30 }, // h_L2
        { 0.20, 0.20, 0.20, 0.20 }, // h_L3
        { 0.00 }                    // o_L
    };
    private static Double[][][] WEIGHTS = {
    //      W0_0  W0_1  W0_2  W0_3      W1_0  W1_1  W1_2  W1_3      W2_0  W2_1  W2_2  W2_3     W3_0  W3_1  W3_2  W3_3
        { { 0.10, 0.20, 0.30, 0.25 }, { 0.40, 0.10, 0.20, 0.15 }, { 0.30, 0.40, 0.50, 0.60 } },                                                        // i_L
        { { 0.20, 0.10, 0.30, 0.25 }, { 0.15, 0.35, 0.25, 0.10 }, { 0.10, 0.20, 0.15, 0.30 }, { 0.25, 0.40, 0.20, 0.20 } }, // h_L0
        { { 0.30, 0.25, 0.15, 0.10 }, { 0.10, 0.35, 0.20, 0.25 }, { 0.25, 0.10, 0.30, 0.15 }, { 0.20, 0.25, 0.10, 0.35 } }, // h_L1
        { { 0.20, 0.15, 0.25, 0.30 }, { 0.25, 0.30, 0.10, 0.20 }, { 0.35, 0.25, 0.20, 0.10 }, { 0.10, 0.20, 0.30, 0.25 } }, // h_L2
        { { 0.30 },                   { 0.20 },                   { 0.25 },                   { 0.35 } }                    // o_L
    };
    private static Integer[] ACTIVATION_FUNCTIONS = {
        0, // i_L
        1, // h_Ls
        1  // o_L
    };
    private static Double LEARNING_RATE = 0.1;
    private static Integer EPOCHS = 100000;
    private static Integer BATCH_SIZE = 3;
    private static Integer NUMBER_OF_INPUTS = BIASES[0].length;
    private static Integer NUMBER_OF_OUTPUTS = BIASES[BIASES.length - 1].length;
    private static Integer NUMBER_OF_HIDDEN_LAYERS = BIASES.length - 2;
    private static Integer NEURONS_PER_HIDDEN_LAYER = BIASES[1].length;
    private static Integer TOTAL_NUMBER_OF_LAYERS = BIASES.length;
    private static String CONFIG_FILE_PATH = "config.txt";
    private static Double[][] TRAINING_INPUTS = {
        // Random training inputs between 0.0 and 1.0
        {0.32, 0.72, 0.90},
        {0.21, 0.15, 0.45},
        {0.67, 0.88, 0.12},
        {0.05, 0.25, 0.75},
        {0.80, 0.60, 0.40},
        {0.50, 0.50, 0.50},
        {0.10, 0.90, 0.30},
        {0.95, 0.05, 0.85},
        {0.40, 0.20, 0.60},
        {0.70, 0.30, 0.10},
        {0.55, 0.65, 0.75},
        {0.85, 0.15, 0.25},
        {0.60, 0.40, 0.20},
        {0.30, 0.80, 0.70},
        {0.25, 0.35, 0.95},
        {0.45, 0.55, 0.05},
        {0.78, 0.22, 0.88},
        {0.33, 0.66, 0.99},
        {0.11, 0.44, 0.77},
        {0.99, 0.01, 0.49},
        {0.48, 0.58, 0.68},
        {0.29, 0.39, 0.59},
        {0.61, 0.71, 0.81},
        {0.14, 0.24, 0.34},
        {0.74, 0.84, 0.94},
        {0.23, 0.43, 0.63},
        {0.57, 0.67, 0.77},
        {0.38, 0.48, 0.58},
        {0.92, 0.82, 0.72},
        {0.16, 0.26, 0.36},
        {0.69, 0.79, 0.89},
        {0.41, 0.51, 0.61},
        {0.53, 0.63, 0.73},
        {0.87, 0.97, 0.07},
        {0.34, 0.54, 0.64},
        {0.76, 0.86, 0.96},
        {0.18, 0.28, 0.38},
        {0.65, 0.75, 0.85},
        {0.42, 0.52, 0.62},
        {0.88, 0.98, 0.08},
        {0.27, 0.37, 0.47},
        {0.59, 0.69, 0.79},
        {0.13, 0.23, 0.33},
        {0.73, 0.83, 0.93},
        {0.31, 0.41, 0.51},
        {0.54, 0.64, 0.74},
        {0.81, 0.91, 0.11},
        {0.22, 0.32, 0.42},
        {0.68, 0.78, 0.88},
        {0.46, 0.56, 0.66},
        {0.15, 0.25, 0.35},
        {0.75, 0.85, 0.95},
        {0.39, 0.49, 0.59},
        {0.91, 0.81, 0.71},
        {0.17, 0.27, 0.37},
        {0.64, 0.74, 0.84},
        {0.43, 0.53, 0.63},
        {0.86, 0.96, 0.06},
        {0.26, 0.36, 0.46},
        {0.58, 0.68, 0.78},
        {0.12, 0.22, 0.32},
        {0.72, 0.82, 0.92},
        {0.35, 0.45, 0.55},
        {0.77, 0.87, 0.97},
        {0.19, 0.29, 0.39},
        {0.66, 0.76, 0.86},
        {0.44, 0.54, 0.64},
        {0.89, 0.99, 0.09},
        {0.28, 0.38, 0.48},
        {0.60, 0.70, 0.80},
        {0.14, 0.24, 0.34},
        {0.74, 0.84, 0.94},
        {0.33, 0.43, 0.53},
        {0.55, 0.65, 0.75},
        {0.83, 0.93, 0.03},
        {0.21, 0.31, 0.41},
        {0.69, 0.79, 0.89},
        {0.47, 0.57, 0.67}
    };
    private static Double[][] TRAINING_OUTPUTS;

    /*
     * METHODS
     */

    public static void generateTrainingOutput() {
        // Generate training outputs as XOR of the three inputs
        TRAINING_OUTPUTS = new Double[TRAINING_INPUTS.length][1];
        for (int i = 0; i < TRAINING_INPUTS.length; i++) {
            Double input0 = TRAINING_INPUTS[i][0];
            Double input1 = TRAINING_INPUTS[i][1];
            Double input2 = TRAINING_INPUTS[i][2];
            // XOR logic: output is 1.0 if an odd number of inputs are > 0.5, else 0.0
            int count = 0;
            if (input0 > 0.5) count++;
            if (input1 > 0.5) count++;
            if (input2 > 0.5) count++;
            TRAINING_OUTPUTS[i][0] = (count % 2 == 1) ? 1.0 : 0.0;
        }
    }
    
    /*
     * GETTERS AND SETTERS
     */

    public static Double[] getInputValues() { return INPUT_VALUES; }
    public static void setInputValues(Double[] inputValues) { INPUT_VALUES = inputValues; }
    public static Double[][] getBiases() { return BIASES; }
    public static void setBiases(Double[][] biases) { BIASES = biases; }
    public static Double[][][] getWeights() { return WEIGHTS; }
    public static void setWeights(Double[][][] weights) { WEIGHTS = weights; }
    public static Integer[] getActivationFunctions() { return ACTIVATION_FUNCTIONS; }
    public static void setActivationFunctions(Integer[] activationFunctions) { ACTIVATION_FUNCTIONS = activationFunctions; }
    public static Double getLearningRate() { return LEARNING_RATE; }
    public static void setLearningRate(Double learningRate) { LEARNING_RATE = learningRate; }
    public static Integer getEpochs() { return EPOCHS; }
    public static void setEpochs(Integer epochs) { EPOCHS = epochs; }
    public static Integer getBatchSize() { return BATCH_SIZE; }
    public static void setBatchSize(Integer batchSize) { BATCH_SIZE = batchSize; }
    public static Integer getNumberOfInputs() { return NUMBER_OF_INPUTS; }
    public static Integer getNumberOfOutputs() { return NUMBER_OF_OUTPUTS; }
    public static Integer getNumberOfHiddenLayers() { return NUMBER_OF_HIDDEN_LAYERS; }
    public static Integer getNeuronsPerHiddenLayer() { return NEURONS_PER_HIDDEN_LAYER; }
    public static Integer getTotalNumberOfLayers() { return TOTAL_NUMBER_OF_LAYERS; }
    public static String getConfigFilePath() { return CONFIG_FILE_PATH; }
    public static Double[][] getTrainingInputs() { return TRAINING_INPUTS; }
    public static void setTrainingInputs(Double[][] trainingInputs) { TRAINING_INPUTS = trainingInputs; }
    public static Double[][] getTrainingOutputs() { return TRAINING_OUTPUTS; }
    public static void setTrainingOutputs(Double[][] trainingOutputs) { TRAINING_OUTPUTS = trainingOutputs; }
}
