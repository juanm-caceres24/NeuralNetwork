package src.models;

import java.util.ArrayList;

import src.activationfunction.ActivationFunction;

public class Neuron {

    /*
     * ATTRIBUTES
     */

    private Integer neuronId;
    private Double value;
    private Double bias;
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
        value = 0.0;
        this.bias = bias;
        this.activationFunction = activationFunction;
        this.forwardWeights = forwardWeights;
        this.backwardWeights = backwardWeights;
    }

    /*
     * METHODS
     */

    public void calcuateValue(ArrayList<Double> inputs) {
        Double sum = 0.0;
        for (int i = 0; i < inputs.size(); i++) {
            sum += inputs.get(i) * this.backwardWeights.get(i);
        }
        this.value = sum + this.bias;
    }

    public void applyActivationFunction() {
        this.value = this.activationFunction.activate(this.value);
    }

    /*
     * GETTERS AND SETTERS
     */

    public Integer getNeuronId() { return neuronId; }

    public Double getValue() { return value; }

    public void setValue(Double value) { this.value = value; }

    public Double getBias() { return bias; }

    public void setBias(Double bias) { this.bias = bias; }

    public ActivationFunction getActivationFunction() { return activationFunction; }

    public void setActivationFunction(ActivationFunction activationFunction) { this.activationFunction = activationFunction; }

    public ArrayList<Double> getForwardWeights() { return forwardWeights; }

    public void setForwardWeights(ArrayList<Double> forwardWeights) { this.forwardWeights = forwardWeights; }

    public ArrayList<Double> getBackwardWeights() { return backwardWeights; }

    public void setBackwardWeights(ArrayList<Double> backwardWeights) { this.backwardWeights = backwardWeights; }
}
