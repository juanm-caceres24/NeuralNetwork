package src.user_interface.impl;

import java.util.ArrayList;

import src.LayerFile;
import src.Layer;
import src.Neuron;
import src.user_interface.UserInterface;

public class Console implements UserInterface {
    
    @Override
    public String createOrLoadNeuralNetwork() {
        return System.console().readLine("CREATE 'c' or LOAD (default) neural network: ");
    }

    @Override
    public ArrayList<LayerFile> createLayerFile(Double INIT_BIAS, Double INIT_WEIGHT) {
        ArrayList<LayerFile> layersFile = new ArrayList<>();
        LayerFile layerFile = new LayerFile();
        String activationFunction;
        int maxNeurons;
        int maxLayers;
        int previousMaxNeurons = 0;

        // number of neurons in the input layer
        maxNeurons = Integer.parseInt(System.console().readLine("Number of neurons in the input layer: "));

        // activation function in the input layer
        layerFile.setActivationFunction(System.console().readLine("Activation function of the input layer: "));

        // load each neuron of the input layer
        for (int i = 0; i < maxNeurons; i++) {
            layerFile.biases.add(INIT_BIAS);
            layerFile.previousWeights = new ArrayList<>();
        }
        previousMaxNeurons = maxNeurons;

        // load the input layer layer into setupFile
        layersFile.add(layerFile);

        // number of neurons per layer in hidden layers
        maxNeurons = Integer.parseInt(System.console().readLine("Number of neurons per layer in hidden layers: "));

        // number of hidden layers
        maxLayers = Integer.parseInt(System.console().readLine("Number of hidden layers: "));

        // activation function in the hidden layers
        activationFunction = System.console().readLine("Activation function of the hidden layer: ");

        // load each hidden layer
        for (int i = 0; i < maxLayers; i++) {
            layerFile = new LayerFile();

            // activation function in the hidden layer
            layerFile.setActivationFunction(activationFunction);

            // load each neuron of the hidden layer
            for (int j = 0; j < maxNeurons; j++) {
                layerFile.biases.add(INIT_BIAS);
                layerFile.previousWeights.add(new ArrayList<>());
                for (int k = 0; k < previousMaxNeurons; k++) {
                    layerFile.previousWeights.get(j).add(INIT_WEIGHT);
                }
            }

            // load the input layer layer into setupFile
            layersFile.add(layerFile);
            previousMaxNeurons = maxNeurons;
        }
        layerFile = new LayerFile();

        // number of neurons in the output layer
        maxNeurons = Integer.parseInt(System.console().readLine("Number of neurons in the output layer: "));

        // activation function in the output layer
        layerFile.setActivationFunction(System.console().readLine("Activation function of the output layer: "));

        // load each neuron of the output layer
        for (int i = 0; i < maxNeurons; i++) {
            layerFile.biases.add(INIT_BIAS);
            layerFile.previousWeights.add(new ArrayList<>());
            for (int j = 0; j < previousMaxNeurons; j++) {
                layerFile.previousWeights.get(i).add(INIT_WEIGHT);
            }
        }

        // load the output layer layer into setupFile
        layersFile.add(layerFile);
        return layersFile;
    }

    @Override
    public void showLayers(ArrayList<Layer> layers) {
        int i = 0;
        for (Layer layer : layers) {
            System.out.printf("LAYER %d:\n", i);
            showLayer(layer);
            i++;
        }
    }

    @Override
    public void showLayer(Layer layer) {
            for (Neuron neuron : layer.getNeurons()) {
                System.out.printf("| %10f | ", neuron.getValue());
            }
            System.out.println();
            for (Neuron neuron : layer.getNeurons()) {
                System.out.printf("| %10f | ", neuron.getBias());
            }
            System.out.println();
            for (Neuron neuron : layer.getNeurons()) {
                System.out.printf("| %10s | ", neuron.getActivationFunction().getClass().getSimpleName());
            }
            System.out.println();
            for (Neuron neuron : layer.getNeurons()) {
                System.out.printf("| %10s | ", neuron.getNeuronType());
            }
            System.out.println();
            int max = layer.getNeurons().get(0).getPreviousNeurons().size();
            if (max == 0) {
                for (int j = 0; j < layer.getNeurons().size(); j++) {
                    System.out.printf("|       void | ");
                }
                System.out.println();
            }
            for (int i = 0; i < max; i++) {
                for (Neuron neuron : layer.getNeurons()) {
                    System.out.printf("| %10f | ", neuron.getPreviousNeurons().get(i).weight);
                }
                System.out.println();
            }
    }

    @Override
    public void clearConsole() {
        System.out.print("\033[H\033[2J");
        System.out.flush();
    }
}
