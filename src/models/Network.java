package src.models;

import java.util.ArrayList;

import src.Setup;
import src.activation_function.ActivationFunction;
import src.activation_function.impl.None;
import src.activation_function.impl.ReLu;
import src.activation_function.impl.Sigmoid;
import src.activation_function.impl.TanH;

public class Network {

    /*
     * ATTRIBUTES
     */

    private ArrayList<Layer> layers;
    private ArrayList<Neuron> neurons;
    
    // Setup parameters
    private Double[] inputValues;
    private Double[][] biases;
    private Double[][][] weights;
    private Integer[] activationFunctions;
    private Integer numberOfInputs;
    private Integer numberOfOutputs;
    private Integer numberOfHiddenLayers;
    private Integer neuronsPerHiddenLayer;
    private Integer totalNumberOfLayers;

    /*
     * CONSTRUCTORS
     */

    public Network() {
        this.layers = new ArrayList<>();
        this.neurons = new ArrayList<>();
        this.inputValues = Setup.getInputValues();
        this.biases = Setup.getBiases();
        this.weights = Setup.getWeights();
        this.activationFunctions = Setup.getActivationFunctions();
        this.numberOfInputs = Setup.getNumberOfInputs();
        this.numberOfOutputs = Setup.getNumberOfOutputs();
        this.numberOfHiddenLayers = Setup.getNumberOfHiddenLayers();
        this.neuronsPerHiddenLayer = Setup.getNeuronsPerHiddenLayer();
        this.totalNumberOfLayers = Setup.getTotalNumberOfLayers();
        this.createNetwork();
    }

    /*
     * METHODS
     */

    public void createNetwork() {
        Integer neuronId = 0;
        Integer layerId = 0;
        Layer previousLayer = null;
        // Create and add layers
        for (int i = 0; i < this.totalNumberOfLayers; i++) {
            ArrayList<Neuron> neurons = new ArrayList<>();
            // Create and add neurons to the layer
            for (int j = 0; j < this.biases[i].length; j++) {
                ArrayList<Double> forwardWeights = (i < this.totalNumberOfLayers - 1) ? new ArrayList<Double>() : null;
                ArrayList<Double> backwardWeights = (i > 0) ? new ArrayList<Double>() : null;
                if (i < this.totalNumberOfLayers - 1) {
                    for (int k = 0; k < this.weights[i][j].length; k++) {
                        forwardWeights.add(this.weights[i][j][k]);
                    }
                }
                if (i > 0) {
                    for (int k = 0; k < this.weights[i - 1].length; k++) {
                        backwardWeights.add(this.weights[i - 1][k][j]);
                    }
                }
                Neuron neuronTmp = new Neuron(neuronId, this.biases[i][j], this.mapActivationFunction(this.activationFunctions[i]), forwardWeights, backwardWeights
                );
                this.neurons.add(neuronTmp);
                neurons.add(neuronTmp);
                neuronId++;
            }
            Layer layer = new Layer(layerId, neurons, null, previousLayer);
            layers.add(layer);
            layerId++;
            previousLayer = layer;
        }
        // Set the next layer for each layer
        for (int i = 0; i < layers.size() - 1; i++) {
            layers.get(i).setNextLayer(layers.get(i + 1));
        }
    }

    public ActivationFunction mapActivationFunction(Integer functionType) {
        switch (functionType) {
            case 0:
                return new None();
            case 1:
                return new Sigmoid();
            case 2:
                return new ReLu();
            case 3:
                return new TanH();
            default:
                // Default to None if unknown
                return new None(); 
        }
    }

    public Integer mapActivationFunction(ActivationFunction function) {
        switch (function.getClass().getSimpleName()) {
            case "None":
                return 0;
            case "Sigmoid":
                return 1;
            case "ReLu":
                return 2;
            case "TanH":
                return 3;
            default:
                // Default to None if unknown
                return 0;
        }
    }

    public void saveNetwork() {
        // Gets the current weights and biases from the network and saves them into the Network attributes
        for (int i = 0; i < layers.size(); i++) {
            Layer layer = layers.get(i);
            // Save biases
            for (int j = 0; j < layer.getNeurons().size(); j++) {
                Neuron neuron = layer.getNeurons().get(j);
                this.biases[i][j] = neuron.getBias();
            }
            // Save weights
            if (i < layers.size() - 1) { // skip output layer
                for (int j = 0; j < layer.getNeurons().size(); j++) {
                    Neuron neuron = layer.getNeurons().get(j);
                    for (int k = 0; k < neuron.getForwardWeights().size(); k++) {
                        this.weights[i][j][k] = neuron.getForwardWeights().get(k);
                    }
                }
            }
        }
        // Save activation functions
        for (int i = 0; i < layers.size(); i++) {
            Layer layer = layers.get(i);
            if (i != 0 && i != layers.size() - 1) {
                this.activationFunctions[i] = this.mapActivationFunction(layer.getNeurons().get(0).getActivationFunction());
            } else if (i == layers.size() - 1) {
                this.activationFunctions[i] = this.mapActivationFunction(layer.getNeurons().get(0).getActivationFunction());
            } else {
                this.activationFunctions[i] = this.mapActivationFunction(layer.getNeurons().get(0).getActivationFunction());
            }
        }
    }

    public void predict() {
        forward(this.inputValues);
    }

    public void forward(Double[] inputValues) {
        // Load input values into the input layer
        for (Neuron neuron : this.layers.get(0).getNeurons()) {
            neuron.setValue(inputValues[neuron.getNeuronId()]);
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
    public Double[] getInputValues() { return inputValues; }
    public void setInputValues(Double[] inputValues) { this.inputValues = inputValues; }
    public Double[][] getBiases() { return biases; }
    public Double[][][] getWeights() { return weights; }
    public Integer[] getActivationFunctions() { return activationFunctions; }
    public Integer getNumberOfInputs() { return numberOfInputs; }
    public Integer getNumberOfOutputs() { return numberOfOutputs; }
    public Integer getNumberOfHiddenLayers() { return numberOfHiddenLayers; }
    public Integer getNeuronsPerHiddenLayer() { return neuronsPerHiddenLayer; }
    public Integer getTotalNumberOfLayers() { return totalNumberOfLayers; }
}
