package src;
import java.util.ArrayList;

import src.activation_functions.ActivationFunction;
import src.activation_functions.impl.None;
import src.activation_functions.impl.Sigmoid;

public class LayerSetupFileStruct {
    
    public ArrayList<Double> biases;
    public ArrayList<ArrayList<Double>> previousWeights;
    public ActivationFunction activationFunction;

    public LayerSetupFileStruct() {
        biases = null;
        previousWeights = null;
        activationFunction = null;
    }

    public LayerSetupFileStruct(ArrayList<Double> biases, ArrayList<ArrayList<Double>> previousWeights, String strActivationFunction) {
        this.biases = biases;
        this.previousWeights = previousWeights;
        this.activationFunction = setActivationFunction(strActivationFunction);
    }

    private ActivationFunction setActivationFunction(String strActivationFunction) {
        switch (strActivationFunction) {
            case "SIGMOID":
                return new Sigmoid();
            default:
                return new None();
        }
    }
}
