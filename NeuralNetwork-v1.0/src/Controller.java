package src;

import java.util.ArrayList;

public class Controller implements Runnable {

    static ArrayList<LayerSetupStruct> layersSetup;
    static FileReadWrite fileRW;

    public Controller() {
        layersSetup = new ArrayList<>();
        fileRW = new FileReadWrite();
    }

    @Override
    public void run() {
        loadNeuralNetworkFromFile();
        showNeuralNetwork();
        while (true) {

        }
    }

    private void loadNeuralNetworkFromFile() {
        // arraylist of layersSetup readed from file, layers formed by an <int> and a <ActivationFunction>
        ArrayList<LayerSetupFileStruct> setupFile = fileRW.readNeuralNetworkFromFile();
        // auxiliary neuron variable to load
        Neuron newNeuron;
        // new arraylist of NeuronConnections to load the current neural layer with the previous neural layer
        ArrayList<NeuronConnectionStruct> previousNeuronConnections = new ArrayList<>();
        // index of actual layer
        int i = 0;
        for (LayerSetupFileStruct layerSetupFile : setupFile) {
            // new layer that will be loaded with neurons
            LayerSetupStruct layer = new LayerSetupStruct();
            // new auxiliary arraylist of NeuronConnections to load the current neural layer of the next iteration with the previous neural layer
            ArrayList<NeuronConnectionStruct> nextPreviousNeuronConnections = new ArrayList<>();
            // loop to create all the neurons in the layer
            for (int j = 0; j < layerSetupFile.biases.size(); j++) {
                // load the layer 0
                if (i == 0) {
                    // create a new neuron in the current layer
                    newNeuron = new Neuron(layerSetupFile.biases.get(j), layerSetupFile.activationFunction, null);
                } else {
                    // create a new neuron in the current layer
                    newNeuron = new Neuron(layerSetupFile.biases.get(j), layerSetupFile.activationFunction, previousNeuronConnections);
                }
                    // add the current neuron to the current layer
                    layer.add(newNeuron);
                    // load the current neuron and weight to load a connection to the following layer
                    nextPreviousNeuronConnections.add(new NeuronConnectionStruct(newNeuron, layerSetupFile.previousWeights.get(j).get(i)));
            }
            previousNeuronConnections = nextPreviousNeuronConnections;
            i++;
            // add the current layer to the layerSetup
            layersSetup.add(layer);
        }
    }

    private void showNeuralNetwork() {

    }
}
