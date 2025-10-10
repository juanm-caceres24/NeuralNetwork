package src.models;

import java.util.ArrayList;

import src.Setup;
import src.activationfunction.ActivationFunction;
import src.activationfunction.impl.None;
import src.activationfunction.impl.ReLu;
import src.activationfunction.impl.Sigmoid;
import src.activationfunction.impl.TanH;

public class Network {

    /*
     * ATTRIBUTES
     */

    private ArrayList<Layer> layers;

    /*
     * CONSTRUCTORS
     */

    public Network() {
        this.layers = new ArrayList<>();
        this.initializeNetwork();
    }

    /*
     * METHODS
     */

    public void initializeNetwork() {
        Integer activationFunctionIndex;
        Integer neuronId = 0;
        Integer layerId = 0;
        Layer previousLayer = null;
        // Create and add layers
        for (int i = 0; i < Setup.getTotalNumberOfLayers(); i++) {
            if (i != 0 && i != Setup.getTotalNumberOfLayers() - 1) {
                activationFunctionIndex = 1;
            } else if (i == Setup.getTotalNumberOfLayers() - 1) {
                activationFunctionIndex = 2;
            } else {
                activationFunctionIndex = 0;
            }
            ArrayList<Neuron> neurons = new ArrayList<>();
            // Create and add neurons to the layer
            for (int j = 0; j < Setup.getBiases()[i].length; j++) {
                ArrayList<Double> forwardWeights = (i < Setup.getTotalNumberOfLayers() - 1) ? new ArrayList<Double>() : null;
                ArrayList<Double> backwardWeights = (i > 0) ? new ArrayList<Double>() : null;
                if (i < Setup.getTotalNumberOfLayers() - 1) {
                    for (int k = 0; k < Setup.getWeights()[i][j].length; k++) {
                        forwardWeights.add(Setup.getWeights()[i][j][k]);
                    }
                }
                if (i > 0) {
                    for (int k = 0; k < Setup.getWeights()[i - 1].length; k++) {
                        backwardWeights.add(Setup.getWeights()[i - 1][k][j]);
                    }
                }
                neurons.add(new Neuron(
                    neuronId,
                    Setup.getBiases()[i][j],
                    this.getActivationFunction(Setup.getActivationFunctions()[activationFunctionIndex]),
                    forwardWeights,
                    backwardWeights
                ));
                neuronId++;
            }
            Layer layer = new Layer(
                    layerId,
                    neurons,
                    null,
                    previousLayer
            );
            layers.add(layer);
            layerId++;
            previousLayer = layer;
        }
        // Set the next layer for each layer
        for (int i = 0; i < layers.size() - 1; i++) {
            layers.get(i).setNextLayer(layers.get(i + 1));
        }
    }

    public ActivationFunction getActivationFunction(Integer functionType) {
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
    
    /*
     * GETTERS AND SETTERS
     */

    public ArrayList<Layer> getLayers() { return layers; }
}
