package src;
import java.util.ArrayList;
import src.activation_functions.ActivationFunction;

public class Neuron implements Runnable {

    private Double value;
    private Double bias;
    private ActivationFunction activationFunction;
    private ArrayList<NeuronConnectionStruct> previousNeurons;

    public Neuron(Double bias, ActivationFunction activationFunction, ArrayList<NeuronConnectionStruct> previousNeurons) {
        value = 0.0;
        this.bias = bias;
        this.activationFunction = activationFunction;
        this.previousNeurons = previousNeurons;
    }

    @Override
    public void run() {
        while (true) {
            
        }
    }

    private void calculateValue() {
        value = 0.0;
        for (NeuronConnectionStruct connection : previousNeurons) {
            value += connection.neuron.getValue() * connection.weight;
        }
        value += bias;
        value = activationFunction.applyActivationFunction(value);
    }

    public Double getValue() {
        return value;
    }
}
