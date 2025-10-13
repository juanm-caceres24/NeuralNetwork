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
        3, // i_L
        5, // h_L0
        5, // h_L1
        5, // h_L2
        2  // o_L
    };
    private static Integer NUMBER_OF_INPUTS = LAYER_SIZES[0];
    private static Integer NUMBER_OF_OUTPUTS = LAYER_SIZES[LAYER_SIZES.length - 1];
    private static Integer NUMBER_OF_HIDDEN_LAYERS = LAYER_SIZES.length - 2;
    private static Integer NEURONS_PER_HIDDEN_LAYER = LAYER_SIZES[1];
    private static Integer TOTAL_NUMBER_OF_LAYERS = LAYER_SIZES.length;
    private static Double[][] BIASES;
    private static Double[][][] WEIGHTS;
    private static Integer[] ACTIVATION_FUNCTIONS = {
        0, // i_L
        1, // h_L0
        1, // h_L1
        1, // h_L2
        1  // o_L
    };

    // Training parameters
    private static Double LEARNING_RATE = 0.1;
    private static Integer EPOCHS = 1000;
    private static Integer BATCH_SIZE = 5;
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

    public static void generateTestTrainingValues() {
        Integer trainingDataLength = 10000;
        TRAINING_DATA = new Double[trainingDataLength][2][NUMBER_OF_INPUTS > NUMBER_OF_OUTPUTS ? NUMBER_OF_INPUTS : NUMBER_OF_OUTPUTS];
        // Generate random n-inputs between 0.0 and 1.0
        for (int i = 0; i < trainingDataLength; i++) {
            Double[] inputs = TRAINING_DATA[i][0];
            for (int j = 0; j < inputs.length; j++) {
                inputs[j] = Math.random();
            }
        }
        // Generate training outputs as XOR of the inputs (set all the outputs to 0.0 or 1.0) (use modular code to n-inputs and m-outputs)
        for (int i = 0; i < trainingDataLength; i++) {
            Double[] inputs = TRAINING_DATA[i][0];
            Double[] outputs = TRAINING_DATA[i][1];
            // XOR logic: output is 1.0 if an odd number of inputs are > 0.5, else 0.0
            Integer count = 0;
            for (Double input : inputs) {
                if (input > 0.5) count++;
            }
            for (int j = 0; j < outputs.length; j++) {
                if (j % 2 == 0) {
                    outputs[j] = (count % 2 == 0) ? 1.0 : 0.0;
                } else {
                    outputs[j] = (count % 2 == 1) ? 1.0 : 0.0;
                }
            }
        }
    }

    public static void initializeFromLayerSizes() {
        // Set the dimension of weights and biases according to LAYER_SIZES and initialize them to 0.0
        BIASES = new Double[TOTAL_NUMBER_OF_LAYERS][];
        WEIGHTS = new Double[TOTAL_NUMBER_OF_LAYERS - 1][][];
        for (int i = 0; i < TOTAL_NUMBER_OF_LAYERS; i++) {
            BIASES[i] = new Double[LAYER_SIZES[i]];
            for (int j = 0; j < LAYER_SIZES[i]; j++) {
                // Initialize all biases to random values between -1.0 and 1.0
                BIASES[i][j] = Math.random() * 2 - 1; 
            }
            if (i < TOTAL_NUMBER_OF_LAYERS - 1) {
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

    public static void initializeFromWeightsAndBiases() {
        // Set LAYER_SIZES according to the dimensions of WEIGHTS and BIASES
        TOTAL_NUMBER_OF_LAYERS = BIASES.length;
        LAYER_SIZES = new Integer[TOTAL_NUMBER_OF_LAYERS];
        for (int i = 0; i < TOTAL_NUMBER_OF_LAYERS; i++) {
            LAYER_SIZES[i] = BIASES[i].length;
        }
        NUMBER_OF_INPUTS = LAYER_SIZES[0];
        NUMBER_OF_OUTPUTS = LAYER_SIZES[LAYER_SIZES.length - 1];
        NUMBER_OF_HIDDEN_LAYERS = LAYER_SIZES.length - 2;
        NEURONS_PER_HIDDEN_LAYER = LAYER_SIZES[1];
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

    public static Integer getNumberOfInputs() { return NUMBER_OF_INPUTS; }
    public static Integer getNumberOfOutputs() { return NUMBER_OF_OUTPUTS; }
    public static Integer getNumberOfHiddenLayers() { return NUMBER_OF_HIDDEN_LAYERS; }
    public static Integer getNeuronsPerHiddenLayer() { return NEURONS_PER_HIDDEN_LAYER; }
    public static Integer getTotalNumberOfLayers() { return TOTAL_NUMBER_OF_LAYERS; }

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

    public static Double[] getInputValues() { return INPUT_VALUES; }
    public static void setInputValues(Double[] inputValues) { INPUT_VALUES = inputValues; }
}
