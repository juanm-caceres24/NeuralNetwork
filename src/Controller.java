package src;

import java.util.ArrayList;

import src.user_interface.UserInterface;
import src.user_interface.impl.Console;

public class Controller implements Runnable {

    static final Double INIT_BIAS = -0.5;
    static final Double INIT_WEIGHT = 1.0;

    static ArrayList<Layer> layers;
    static ArrayList<Thread> layersThreads;

    static FileReadWrite fileRW;
    static UserInterface userInterface;

    public Controller() {
        layers = new ArrayList<>();
        layersThreads = new ArrayList<>();
        fileRW = new FileReadWrite();
        userInterface = new Console();
    }

    @Override
    public void run() {

        // setup of neural network
        userInterface.clearConsole();
        if (userInterface.createOrLoadNeuralNetwork().equals("c")) {
            userInterface.clearConsole();
            initNeuralNetwork(userInterface.createLayerFile(INIT_BIAS, INIT_WEIGHT));
            saveNeuralNetwork();
        } else {
            initNeuralNetwork(fileRW.readNeuralNetworkFromFile());
        }

        // start layerThreads
        for (Layer layer : layers) {
            layersThreads.add(new Thread(layer));
        }
        for (Thread layerThread : layersThreads) {
            layerThread.start();
        }

        // main loop
        while (true) {

            // read input
            readInput(fileRW.readInputFromFile());

            // generate output
            generateOutput();
            userInterface.clearConsole();
            userInterface.showLayers(layers);
        }
    }

    private void initNeuralNetwork(ArrayList<LayerFile> layersFile) {

        // loop all the layers of the layerSetupFile
        int i = 0;
        for (LayerFile layerFile : layersFile) {
            Layer layer = new Layer();

            // loop all the neurons of the current layer
            for (int j = 0; j < layerFile.biases.size(); j++) {
                Neuron newNeuron;
                if (i == 0) {
                    newNeuron = new Neuron(layerFile.biases.get(j), layerFile.activationFunction, "INPUT", new ArrayList<>());
                } else if (i == layersFile.size() - 1) {
                    newNeuron = new Neuron(layerFile.biases.get(j), layerFile.activationFunction, "OUTPUT", new ArrayList<>());
                } else {
                    newNeuron = new Neuron(layerFile.biases.get(j), layerFile.activationFunction, "HIDDEN", new ArrayList<>());
                }
                layer.getNeurons().add(newNeuron);
            }

            // add the current created layer into the system layerSetup
            layers.add(layer);
            i++;
        }

        // load all the connections between neurons
        loadNeuronConnections(layersFile);
    }

    private void saveNeuralNetwork() {
        ArrayList<LayerFile> layersFile = new ArrayList<>();

        // loop all the layers of the system layerSetup
        for (Layer layer : layers) {
            LayerFile layerFile = new LayerFile();

            // save the activation function of the first neuron only
            layerFile.activationFunction = layer.getNeurons().get(0).getActivationFunction();

            // loop all the neurons in the current layer
            int i = 0;
            for (Neuron neuron : layer.getNeurons()) {
                layerFile.previousWeights.add(new ArrayList<>());

                // save the bias of the current neuron
                layerFile.biases.add(neuron.getBias());

                // loop all the previousNeurons connections of the current neuron
                for (NeuronConnection previousWeights : neuron.getPreviousNeurons()) {
                    layerFile.previousWeights.get(i).add(previousWeights.weight);
                }
                i++;
            }

            // add the current created layer into the setupFile
            layersFile.add(layerFile);
        }

        // save into setupFile
        fileRW.saveNeuralNetworkIntoFile(layersFile);
    }

    private void loadNeuronConnections(ArrayList<LayerFile> layersFile) {
        Layer previousLayer = null;
        int i = 0;
        for (Layer layer : layers) {
            if (i != 0) {
                int j = 0;
                for (Neuron neuron : layer.getNeurons()) {
                    int k = 0;
                    for (Neuron auxNeuron : previousLayer.getNeurons()) {
                        neuron.getPreviousNeurons().add(new NeuronConnection(auxNeuron, layersFile.get(i).previousWeights.get(j).get(k)));
                        k++;
                    }
                    j++;
                }
            }
            previousLayer = layer;
            i++;
        }
    }

    private void generateOutput() {
        
        // IMPLEMENT METHOD

    }

    private void readInput(ArrayList<Double> values) {
        int i = 0;
        for (Neuron neuron : layers.get(0).getNeurons()) {
            neuron.setValue(values.get(i));
            i++;
        }
    }
}
