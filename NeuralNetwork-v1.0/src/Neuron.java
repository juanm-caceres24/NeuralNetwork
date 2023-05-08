package src;

import java.util.ArrayList;

import src.activation_function.ActivationFunction;

public class Neuron implements Runnable {

    private Double value;
    private Double bias;
    private ActivationFunction activationFunction;
    private String neuronType;
    private ArrayList<NeuronConnection> previousNeurons;

    public Neuron(Double bias, ActivationFunction activationFunction, String neuronType, ArrayList<NeuronConnection> previousNeurons) {
        value = 0.0;
        this.bias = bias;
        this.activationFunction = activationFunction;
        this.neuronType = neuronType;
        this.previousNeurons = previousNeurons;
    }

    @Override
    public void run() {
        while (true) {
            calculateValue();
        }
    }

    private void calculateValue() {
        if (!neuronType.equals("INPUT")) {
            value = 0.0;
            for (NeuronConnection connection : previousNeurons) {
                value += connection.neuron.getValue() * connection.weight;
            }
            value += bias;
        }
        value = activationFunction.applyActivationFunction(value);
    }

    public void setValue(Double value) {
        this.value = value;
    }

    public Double getValue() {
        return value;
    }

    public Double getBias() {
        return bias;
    }

    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }

    public String getNeuronType() {
        return neuronType;
    }

    public ArrayList<NeuronConnection> getPreviousNeurons() {
        return previousNeurons;
    }
}
