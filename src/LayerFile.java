package src;

import java.util.ArrayList;

import src.activation_function.ActivationFunction;
import src.activation_function.impl.None;
import src.activation_function.impl.Sigmoid;

public class LayerFile {
    
    public ArrayList<Double> biases;
    public ArrayList<ArrayList<Double>> previousWeights;
    public ActivationFunction activationFunction;

    public LayerFile() {
        biases = new ArrayList<>();
        previousWeights = new ArrayList<>();
        activationFunction = null;
    }

    public void setActivationFunction(String strActivationFunction) {
        switch (strActivationFunction) {
            case "SIGMOID": {
                activationFunction = new Sigmoid();
                break;
            }
            default: {
                activationFunction = new None();
            }
        }
    }
}
