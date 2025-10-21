package src.models;

import java.util.ArrayList;

import src.utils.activation_function.ActivationFunction;

public class Neuron {

    /*
     * ATTRIBUTES
     */

    // Neuron identifier
    private Integer neuronId;

    // Neuron parameters
    private Double z;
    private Double activation;
    private Double bias;
    private Double y;
    private Double delta;
    private ActivationFunction activationFunction;

    // Neuron topology
    private ArrayList<Double> forwardWeights;
    private ArrayList<Double> backwardWeights;

    /*
     * CONSTRUCTORS
     */

    public Neuron(Integer neuronId, Double bias, ActivationFunction activationFunction, ArrayList<Double> forwardWeights, ArrayList<Double> backwardWeights) {
        this.neuronId = neuronId;
        this.z = 0.0;
        this.activation = 0.0;
        this.bias = bias;
        this.y = 0.0;
        this.delta = 0.0;
        this.activationFunction = activationFunction;
        this.forwardWeights = forwardWeights;
        this.backwardWeights = backwardWeights;
    }

    /*
     * METHODS
     */

    public Double calculateForward(ArrayList<Double> inputs) {
        z = bias;
        for (int i = 0; i < inputs.size(); i++) {
            z += inputs.get(i) * backwardWeights.get(i);
        }
        return z;
    }

    public Double calculateActivation() {
        activation = activationFunction.activate(z);
        return activation;
    }

    public Double calculateBackward(ArrayList<Double> inputs) {
        y = 0.0;
        for (int i = 0; i < inputs.size(); i++) {
            y += inputs.get(i) * forwardWeights.get(i);
        }
        return y;
    }

    public Double calculateDelta() {
        delta = activationFunction.derivative(z) * y;
        return delta;
    }

    /*
     * GETTERS AND SETTERS
     */

    public Integer getNeuronId() { return neuronId; }
    public Double getZ() { return z; }
    public Double getActivation() { return activation; }
    public void setActivation(Double activation) { this.activation = activation; }
    public Double getBias() { return bias; }
    public void setBias(Double bias) { this.bias = bias; }
    public Double getY() { return y; }
    public Double getDelta() { return delta; }
    public void setDelta(Double delta) { this.delta = delta; }
    public ActivationFunction getActivationFunction() { return activationFunction; }
    public ArrayList<Double> getForwardWeights() { return forwardWeights; }
    public ArrayList<Double> getBackwardWeights() { return backwardWeights; }
}
