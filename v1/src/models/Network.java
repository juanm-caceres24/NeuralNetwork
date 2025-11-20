package v1.src.models;

import v1.src.Setup;
import v1.src.utils.activation_function.ActivationFunction;
import v1.src.utils.activation_function.impl.LeakyReLU;
import v1.src.utils.activation_function.impl.Lineal;
import v1.src.utils.activation_function.impl.ReLU;
import v1.src.utils.activation_function.impl.Sigmoid;
import v1.src.utils.activation_function.impl.TanH;

public class Network {

    /*
     * ATTRIBUTES
     */

    // Network topology
    private Layer[] layers;
    private Neuron[] neurons;

    /*
     * CONSTRUCTORS
     */

    public Network() {
        this.layers = null;
        this.neurons = null;
        this.loadNetwork();
    }

    /*
     * METHODS
     */

    public void loadNetwork() {
        double[][] BIASES = Setup.getBiases();
        double[][][] WEIGHTS = Setup.getWeights();
        int[] ACTIVATION_FUNCTIONS = Setup.getActivationFunctions();
        int neuronId = 0;
        int layerId = 0;
        Layer previousLayer = null;
        // Initialize arrays for layers and neurons
        if (BIASES == null) {
            // Nothing to build
            this.layers = new Layer[0];
            this.neurons = new Neuron[0];
            return;
        }
        this.layers = new Layer[BIASES.length];
        int totalNeurons = 0;
        for (int i = 0; i < BIASES.length; i++) totalNeurons += BIASES[i].length;
        this.neurons = new Neuron[totalNeurons];
        // Create and add layers
        for (int i = 0; i < BIASES.length; i++) {
            Neuron[] neurons = new Neuron[BIASES[i].length];
            // Create and add neurons to the layer
            for (int j = 0; j < BIASES[i].length; j++) {
                double[] forwardWeights = (i < BIASES.length - 1) ? new double[WEIGHTS[i][j].length] : null;
                double[] backwardWeights = (i > 0) ? new double[WEIGHTS[i - 1].length] : null;
                if (i < BIASES.length - 1) {
                    for (int k = 0; k < WEIGHTS[i][j].length; k++) {
                        forwardWeights[k] = WEIGHTS[i][j][k];
                    }
                }
                if (i > 0) {
                    for (int k = 0; k < WEIGHTS[i - 1].length; k++) {
                        backwardWeights[k] = WEIGHTS[i - 1][k][j];
                    }
                }
                Neuron neuronTmp = new Neuron(neuronId, BIASES[i][j], this.mapActivationFunction(ACTIVATION_FUNCTIONS[i]), forwardWeights, backwardWeights);
                neurons[j] = neuronTmp;
                neuronId++;
            }
            Layer layer = new Layer(layerId, neurons, null, previousLayer);
            this.layers[layerId] = layer;
            layerId++;
            previousLayer = layer;
        }
        // Set the next layer for each layer
        for (int i = 0; i < layers.length - 1; i++) {
            this.layers[i].setNextLayer(this.layers[i + 1]);
        }
    }

    public ActivationFunction mapActivationFunction(Integer functionType) {
        switch (functionType) {
            case 0:
                return new Lineal();
            case 1:
                return new Sigmoid();
            case 2:
                return new TanH();
            case 3:
                return new ReLU();
            case 4:
                return new LeakyReLU();
            default:
                // Default to Lineal if unknown
                return new Lineal(); 
        }
    }

    public Integer mapActivationFunction(ActivationFunction function) {
        switch (function.getClass().getSimpleName()) {
            case "Lineal":
                return 0;
            case "Sigmoid":
                return 1;
            case "TanH":
                return 2;
            case "ReLU":
                return 3;
            case "LeakyReLU":
                return 4;
            default:
                // Default to Lineal if unknown
                return 0;
        }
    }

    public void saveNetwork() {
        double[][] BIASES = Setup.getBiases();
        double[][][] WEIGHTS = Setup.getWeights();
        int[] ACTIVATION_FUNCTIONS = Setup.getActivationFunctions();
        // Gets the current weights and biases from the network and saves them into the Network attributes
        for (int i = 0; i < layers.length; i++) {
            Layer layer = layers[i];
            // Save biases
            for (int j = 0; j < layer.getNeurons().length; j++) {
                Neuron neuron = layer.getNeurons()[j];
                BIASES[i][j] = neuron.getBias();
            }
            // Save weights
            if (i < layers.length - 1) { // skip output layer
                for (int j = 0; j < layer.getNeurons().length; j++) {
                    Neuron neuron = layer.getNeurons()[j];
                    for (int k = 0; k < neuron.getForwardWeights().length; k++) {
                        WEIGHTS[i][j][k] = neuron.getForwardWeights()[k];
                    }
                }
            }
        }
        // Save activation functions
        for (int i = 0; i < layers.length; i++) {
            Layer layer = layers[i];
            if (i != 0 && i != layers.length - 1) {
                ACTIVATION_FUNCTIONS[i] = this.mapActivationFunction(layer.getNeurons()[0].getActivationFunction());
            } else if (i == layers.length - 1) {
                ACTIVATION_FUNCTIONS[i] = this.mapActivationFunction(layer.getNeurons()[0].getActivationFunction());
            } else {
                ACTIVATION_FUNCTIONS[i] = this.mapActivationFunction(layer.getNeurons()[0].getActivationFunction());
            }
        }
        Setup.setBiases(BIASES);
        Setup.setWeights(WEIGHTS);
        Setup.setActivationFunctions(ACTIVATION_FUNCTIONS);
    }

    public void predict() {
        double[] INPUT_VALUES = Setup.getInputValues();
        forward(INPUT_VALUES);
    }

    public void forward(double[] inputValues) {
        // Load input values into the input layer
        for (Neuron neuron : this.layers[0].getNeurons()) {
            neuron.setActivation(inputValues[neuron.getNeuronId()]);
        }
        // Propagate values through the network
        for (Layer layer : this.layers) {
            if (layer.getPreviousLayer() != null) {
                layer.feedForward();
            }
        }
    }
    
    /*
     * GETTERS AND SETTERS
     */

    public Layer[] getLayers() { return layers; }
    public Neuron[] getNeurons() { return neurons; }
}
