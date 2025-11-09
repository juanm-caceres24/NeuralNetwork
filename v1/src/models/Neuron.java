package v1.src.models;

import v1.src.utils.activation_function.ActivationFunction;

public class Neuron {

    /*
     * ATTRIBUTES
     */

    // Neuron identifier
    private int neuronId;

    // Neuron parameters
    private double z;
    private double activation;
    private double bias;
    private double y;
    private double delta;
    private ActivationFunction activationFunction;

    // Neuron topology
    private double[] forwardWeights;
    private double[] backwardWeights;

    /*
     * CONSTRUCTORS
     */

    public Neuron(int neuronId, double bias, ActivationFunction activationFunction, double[] forwardWeights, double[] backwardWeights) {
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

    public double calculateForward(double[] inputs) {
        z = bias;
        for (int i = 0; i < inputs.length; i++) {
            z += inputs[i] * backwardWeights[i];
        }
        return z;
    }

    public double calculateActivation() {
        activation = activationFunction.activate(z);
        return activation;
    }

    public double calculateBackward(double[] inputs) {
        y = 0.0;
        for (int i = 0; i < inputs.length; i++) {
            y += inputs[i] * forwardWeights[i];
        }
        return y;
    }

    public double calculateDelta() {
        delta = activationFunction.derivative(z) * y;
        return delta;
    }

    /*
     * GETTERS AND SETTERS
     */

    public int getNeuronId() { return neuronId; }
    public double getZ() { return z; }
    public double getActivation() { return activation; }
    public void setActivation(double activation) { this.activation = activation; }
    public double getBias() { return bias; }
    public void setBias(double bias) { this.bias = bias; }
    public double getY() { return y; }
    public double getDelta() { return delta; }
    public void setDelta(double delta) { this.delta = delta; }
    public ActivationFunction getActivationFunction() { return activationFunction; }
    public double[] getForwardWeights() { return forwardWeights; }
    public double[] getBackwardWeights() { return backwardWeights; }
}
