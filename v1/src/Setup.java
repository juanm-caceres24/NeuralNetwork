package v1.src;

public class Setup {

    /*
     * ATTRIBUTES
     */

    // File paths
    private static String CONFIG_FILE_PATH = "v1/config.txt";
    private static String TRAINING_DATA_FILE_PATH = "v1/training_data.txt";
    private static String INPUT_VALUES_FILE_PATH = "v1/input.txt";
    private static String OUTPUT_VALUES_FILE_PATH = "v1/output.txt";

    // Network parameters
    private static int[] LAYERS_SIZE;
    private static double[][] BIASES;
    private static double[][][] WEIGHTS;
    private static int[] ACTIVATION_FUNCTIONS;

    // Training parameters
    private static double LEARNING_RATE = 0.1;
    private static int EPOCHS = 1000;
    private static int BATCH_SIZE = 10;
    private static int TEST_TRAINING_DATA_LENGTH = 1000;
    private static double[][][] TRAINING_DATA;

    // Input values for prediction
    private static double[] INPUT_VALUES;

    /*
     * CONSTRUCTORS
     */

    private Setup() { }

    /*
     * METHODS
     */

    public static void initializeRandomNetwork() {
        // Set the dimension of weights and biases according to LAYERS_SIZE and initialize them to 0.0
        BIASES = new double[LAYERS_SIZE.length][];
        WEIGHTS = new double[LAYERS_SIZE.length - 1][][];
        for (int i = 0; i < LAYERS_SIZE.length; i++) {
            BIASES[i] = new double[LAYERS_SIZE[i]];
            for (int j = 0; j < LAYERS_SIZE[i]; j++) {
                // Initialize all biases to random values between -1.0 and 1.0
                BIASES[i][j] = Math.random() * 2 - 1; 
            }
            if (i < LAYERS_SIZE.length - 1) {
                WEIGHTS[i] = new double[LAYERS_SIZE[i]][LAYERS_SIZE[i + 1]];
                for (int j = 0; j < LAYERS_SIZE[i]; j++) {
                    for (int k = 0; k < LAYERS_SIZE[i + 1]; k++) {
                        // Initialize all weights to random values between -1.0 and 1.0
                        WEIGHTS[i][j][k] = Math.random() * 2 - 1;
                    }
                }
            }
        }
    }
    
    /*
     * GETTERS AND SETTERS
     */

    public static String getConfigFilePath() { return CONFIG_FILE_PATH; }
    public static String getTrainingDataFilePath() { return TRAINING_DATA_FILE_PATH; }
    public static String getInputValuesFilePath() { return INPUT_VALUES_FILE_PATH; }
    public static String getOutputValuesFilePath() { return OUTPUT_VALUES_FILE_PATH; }

    public static int[] getLayersSize() { return LAYERS_SIZE; }
    public static void setLayersSize(int[] layerSizes) { LAYERS_SIZE = layerSizes; }
    public static double[][] getBiases() { return BIASES; }
    public static void setBiases(double[][] biases) { BIASES = biases; }
    public static double[][][] getWeights() { return WEIGHTS; }
    public static void setWeights(double[][][] weights) { WEIGHTS = weights; }
    public static int[] getActivationFunctions() { return ACTIVATION_FUNCTIONS; }
    public static void setActivationFunctions(int[] activationFunctions) { ACTIVATION_FUNCTIONS = activationFunctions; }

    public static double getLearningRate() { return LEARNING_RATE; }
    public static void setLearningRate(double learningRate) { LEARNING_RATE = learningRate; }
    public static int getEpochs() { return EPOCHS; }
    public static void setEpochs(int epochs) { EPOCHS = epochs; }
    public static int getBatchSize() { return BATCH_SIZE; }
    public static void setBatchSize(int batchSize) { BATCH_SIZE = batchSize; }
    public static double[][][] getTrainingData() { return TRAINING_DATA; }
    public static void setTrainingData(double[][][] trainingData) { TRAINING_DATA = trainingData; }
    public static int getTestTrainingDataLength() { return TEST_TRAINING_DATA_LENGTH; }
    public static void setTestTrainingDataLength(int testTrainingDataLength) { TEST_TRAINING_DATA_LENGTH = testTrainingDataLength; }

    public static double[] getInputValues() { return INPUT_VALUES; }
    public static void setInputValues(double[] inputValues) { INPUT_VALUES = inputValues; }
}
