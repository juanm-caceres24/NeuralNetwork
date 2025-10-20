package src;

public class Setup {

    /*
     * ATTRIBUTES
     */

    // File paths
    private static String CONFIG_FILE_PATH = "config.txt";
    private static String TRAINING_DATA_FILE_PATH = "training_data.txt";
    private static String INPUT_VALUES_FILE_PATH = "input.txt";
    private static String OUTPUT_VALUES_FILE_PATH = "output.txt";

    // Network parameters
    private static Integer[] LAYER_SIZES = { // Used for first initialization of network topology
        4,  // i_L
        10, // h_L0
        10, // h_L1
        16   // o_L
    };
    private static Double[][] BIASES;
    private static Double[][][] WEIGHTS;
    private static Integer[] ACTIVATION_FUNCTIONS = {
        0, // i_L
        2, // h_L0
        2, // h_L1
        1  // o_L
    };

    // Training parameters
    private static Double LEARNING_RATE = 0.1;
    private static Integer EPOCHS = 1000;
    private static Integer BATCH_SIZE = 10;
    private static Integer TEST_TRAINING_DATA_LENGTH = 1000;
    private static Double[][][] TRAINING_DATA;

    // Input values for prediction
    private static Double[] INPUT_VALUES;

    /*
     * CONSTRUCTORS
     */

    private Setup() { }

    /*
     * METHODS
     */

    public static void initializeFromLayerSizes() {
        // Set the dimension of weights and biases according to LAYER_SIZES and initialize them to 0.0
        BIASES = new Double[LAYER_SIZES.length][];
        WEIGHTS = new Double[LAYER_SIZES.length - 1][][];
        for (int i = 0; i < LAYER_SIZES.length; i++) {
            BIASES[i] = new Double[LAYER_SIZES[i]];
            for (int j = 0; j < LAYER_SIZES[i]; j++) {
                // Initialize all biases to random values between -1.0 and 1.0
                BIASES[i][j] = Math.random() * 2 - 1; 
            }
            if (i < LAYER_SIZES.length - 1) {
                WEIGHTS[i] = new Double[LAYER_SIZES[i]][LAYER_SIZES[i + 1]];
                for (int j = 0; j < LAYER_SIZES[i]; j++) {
                    for (int k = 0; k < LAYER_SIZES[i + 1]; k++) {
                        // Initialize all weights to random values between -1.0 and 1.0
                        WEIGHTS[i][j][k] = Math.random() * 2 - 1;
                    }
                }
            }
        }
    }

    public static void initializeFromBiases() {
        // Set LAYER_SIZES according to the dimensions of WEIGHTS and BIASES
        LAYER_SIZES = new Integer[BIASES.length];
        for (int i = 0; i < BIASES.length; i++) {
            LAYER_SIZES[i] = BIASES[i].length;
        }
    }
    
    /*
     * GETTERS AND SETTERS
     */

    public static String getConfigFilePath() { return CONFIG_FILE_PATH; }
    public static String getTrainingDataFilePath() { return TRAINING_DATA_FILE_PATH; }
    public static String getInputValuesFilePath() { return INPUT_VALUES_FILE_PATH; }
    public static String getOutputValuesFilePath() { return OUTPUT_VALUES_FILE_PATH; }

    public static Integer[] getLayerSizes() { return LAYER_SIZES; }
    public static void setLayerSizes(Integer[] layerSizes) { LAYER_SIZES = layerSizes; }
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
    public static Double[][][] getTrainingData() { return TRAINING_DATA; }
    public static void setTrainingData(Double[][][] trainingData) { TRAINING_DATA = trainingData; }
    public static Integer getTestTrainingDataLength() { return TEST_TRAINING_DATA_LENGTH; }
    public static void setTestTrainingDataLength(Integer testTrainingDataLength) { TEST_TRAINING_DATA_LENGTH = testTrainingDataLength; }

    public static Double[] getInputValues() { return INPUT_VALUES; }
    public static void setInputValues(Double[] inputValues) { INPUT_VALUES = inputValues; }
}
