package v1.src.models;

import java.util.ArrayList;

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
    private ArrayList<Layer> layers;
    private ArrayList<Neuron> neurons;

    /*
     * CONSTRUCTORS
     */

    public Network() {
        this.layers = new ArrayList<>();
        this.neurons = new ArrayList<>();
        this.createNetwork();
    }

    /*
     * METHODS
     */

    public void createNetwork() {
        // If biases, weights or activation functions are not defined, initialize a random network
        if (Setup.getBiases() == null || Setup.getWeights() == null || Setup.getActivationFunctions() == null) {
            Setup.initializeRandomNetwork();
        }
        Double[][] BIASES = Setup.getBiases();
        Double[][][] WEIGHTS = Setup.getWeights();
        Integer[] ACTIVATION_FUNCTIONS = Setup.getActivationFunctions();
        Integer neuronId = 0;
        Integer layerId = 0;
        Layer previousLayer = null;
        // Create and add layers
        for (int i = 0; i < BIASES.length; i++) {
            ArrayList<Neuron> neurons = new ArrayList<>();
            // Create and add neurons to the layer
            for (int j = 0; j < BIASES[i].length; j++) {
                ArrayList<Double> forwardWeights = (i < BIASES.length - 1) ? new ArrayList<Double>() : null;
                ArrayList<Double> backwardWeights = (i > 0) ? new ArrayList<Double>() : null;
                if (i < BIASES.length - 1) {
                    for (int k = 0; k < WEIGHTS[i][j].length; k++) {
                        forwardWeights.add(WEIGHTS[i][j][k]);
                    }
                }
                if (i > 0) {
                    for (int k = 0; k < WEIGHTS[i - 1].length; k++) {
                        backwardWeights.add(WEIGHTS[i - 1][k][j]);
                    }
                }
                Neuron neuronTmp = new Neuron(neuronId, BIASES[i][j], this.mapActivationFunction(ACTIVATION_FUNCTIONS[i]), forwardWeights, backwardWeights);
                this.neurons.add(neuronTmp);
                neurons.add(neuronTmp);
                neuronId++;
            }
            Layer layer = new Layer(layerId, neurons, null, previousLayer);
            this.layers.add(layer);
            layerId++;
            previousLayer = layer;
        }
        // Set the next layer for each layer
        for (int i = 0; i < layers.size() - 1; i++) {
            this.layers.get(i).setNextLayer(this.layers.get(i + 1));
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
        Double[][] BIASES = Setup.getBiases();
        Double[][][] WEIGHTS = Setup.getWeights();
        Integer[] ACTIVATION_FUNCTIONS = Setup.getActivationFunctions();
        // Gets the current weights and biases from the network and saves them into the Network attributes
        for (int i = 0; i < layers.size(); i++) {
            Layer layer = layers.get(i);
            // Save biases
            for (int j = 0; j < layer.getNeurons().size(); j++) {
                Neuron neuron = layer.getNeurons().get(j);
                BIASES[i][j] = neuron.getBias();
            }
            // Save weights
            if (i < layers.size() - 1) { // skip output layer
                for (int j = 0; j < layer.getNeurons().size(); j++) {
                    Neuron neuron = layer.getNeurons().get(j);
                    for (int k = 0; k < neuron.getForwardWeights().size(); k++) {
                        WEIGHTS[i][j][k] = neuron.getForwardWeights().get(k);
                    }
                }
            }
        }
        // Save activation functions
        for (int i = 0; i < layers.size(); i++) {
            Layer layer = layers.get(i);
            if (i != 0 && i != layers.size() - 1) {
                ACTIVATION_FUNCTIONS[i] = this.mapActivationFunction(layer.getNeurons().get(0).getActivationFunction());
            } else if (i == layers.size() - 1) {
                ACTIVATION_FUNCTIONS[i] = this.mapActivationFunction(layer.getNeurons().get(0).getActivationFunction());
            } else {
                ACTIVATION_FUNCTIONS[i] = this.mapActivationFunction(layer.getNeurons().get(0).getActivationFunction());
            }
        }
        Setup.setBiases(BIASES);
        Setup.setWeights(WEIGHTS);
        Setup.setActivationFunctions(ACTIVATION_FUNCTIONS);
    }

    public void predict() {
        Double[] INPUT_VALUES = Setup.getInputValues();
        forward(INPUT_VALUES);
    }

    public void forward(Double[] inputValues) {
        // Load input values into the input layer
        for (Neuron neuron : this.layers.get(0).getNeurons()) {
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

    public ArrayList<Layer> getLayers() { return layers; }
    public ArrayList<Neuron> getNeurons() { return neurons; }
}
