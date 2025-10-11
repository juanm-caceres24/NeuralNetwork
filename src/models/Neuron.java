package src.models;

import java.util.ArrayList;

import src.activation_function.ActivationFunction;

public class Neuron {

    /*
     * ATTRIBUTES
     */

    private Integer neuronId;
    private Double z;
    private Double value;
    private Double bias;
    private Double delta;
    private ActivationFunction activationFunction;
    private ArrayList<Double> forwardWeights;
    private ArrayList<Double> backwardWeights;

    /*
     * CONSTRUCTORS
     */

    public Neuron(
            Integer neuronId,
            Double bias,
            ActivationFunction activationFunction,
            ArrayList<Double> forwardWeights,
            ArrayList<Double> backwardWeights) {

        this.neuronId = neuronId;
        this.z = 0.0;
        value = 0.0;
        this.bias = bias;
        this.delta = 0.0;
        this.activationFunction = activationFunction;
        this.forwardWeights = forwardWeights;
        this.backwardWeights = backwardWeights;
    }

    /*
     * METHODS
     */

    public void calculateValue(ArrayList<Double> inputs) {
        z = bias;
        for (int i = 0; i < inputs.size(); i++) {
            z += inputs.get(i) * backwardWeights.get(i);
        }
        value = activationFunction.activate(z);
    }

    public void calculateDelta(ArrayList<Double> inputs) {
        Double sum = 0.0;
        for (int i = 0; i < inputs.size(); i++) {
            sum += inputs.get(i) * forwardWeights.get(i);
        }
        delta = activationFunction.derivative(z) * sum;
    }

    /*
     * GETTERS AND SETTERS
     */

    public Integer getNeuronId() { return neuronId; }
    public Double getZ() { return z; }
    public void setZ(Double z) { this.z = z; }
    public Double getValue() { return value; }
    public void setValue(Double value) { this.value = value; }
    public Double getBias() { return bias; }
    public void setBias(Double bias) { this.bias = bias; }
    public Double getDelta() { return delta; }
    public void setDelta(Double delta) { this.delta = delta; }
    public ActivationFunction getActivationFunction() { return activationFunction; }
    public void setActivationFunction(ActivationFunction activationFunction) { this.activationFunction = activationFunction; }
    public ArrayList<Double> getForwardWeights() { return forwardWeights; }
    public void setForwardWeights(ArrayList<Double> forwardWeights) { this.forwardWeights = forwardWeights; }
    public ArrayList<Double> getBackwardWeights() { return backwardWeights; }
    public void setBackwardWeights(ArrayList<Double> backwardWeights) { this.backwardWeights = backwardWeights; }
}
