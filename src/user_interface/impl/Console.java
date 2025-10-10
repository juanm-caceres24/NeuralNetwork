package src.user_interface.impl;

import src.models.Layer;
import src.models.Network;
import src.models.Neuron;
import src.user_interface.UserInterface;

public class Console implements UserInterface {

    /*
     * ATTRIBUTES
     */

    private static Network network;

    /*
     * CONSTRUCTORS
     */

    public Console(Network network) {
        Console.network = network;
    }

    /*
     * METHODS
     */

    @Override
    public void showError(Integer errorCode) {
        switch (errorCode) {
            case 1:
                System.out.println("Error: Invalid input.");
                break;
            case 2:
                System.out.println("Error: File not found.");
                break;
            case 3:
                System.out.println("Error: Unable to save file.");
                break;
            default:
                System.out.println("Error: Unknown error.");
                break;
        }
    }

    @Override
    public void showNetwork() {
        System.out.println("========================================|");
        System.out.println(" NETWORK                                |");
        System.out.println("========================================|");
        for (Layer layer : network.getLayers()) {
            System.out.println("                                        |");
            System.out.println(" >> | Layer Id ------------------------ | " + layer.getLayerId());
            if (layer.getNextLayer() == null) {
                System.out.println("    | Next Layer Id ------------------- | (none)");
            } else {
                System.out.println("    | Next Layer Id ------------------- | " + layer.getNextLayer().getLayerId());
            }
            if (layer.getPreviousLayer() == null) {
                System.out.println("    | Previous Layer Id --------------- | (none)");
            } else {
                System.out.println("    | Previous Layer Id --------------- | " + layer.getPreviousLayer().getLayerId());
            }
            for (Neuron neuron : layer.getNeurons()) {
                System.out.println("    +                                   |");
                System.out.println("    | ->> | Neuron Id ----------------- | " + neuron.getNeuronId());
                System.out.println("    |     | Value --------------------- | " + neuron.getValue());
                System.out.println("    |     | Bias ---------------------- | " + neuron.getBias());
                System.out.println("    |     | Activation Function ------- | " + neuron.getActivationFunction().getClass().getSimpleName());
                if (neuron.getForwardWeights() != null) {
                    System.out.println("    |     | Forward Weights ----------- | " + neuron.getForwardWeights());
                } else {
                    System.out.println("    |     | Forward Weights ----------- | (none)");
                }
                if (neuron.getBackwardWeights() != null) {
                    System.out.println("    |     | Backward Weights ---------- | " + neuron.getBackwardWeights());
                } else {
                    System.out.println("    |     | Backward Weights ---------- | (none)");
                }
            }
        }
        System.out.println("                                        |");
    }

    @Override
    public void showInputs() {
        System.out.println("========================================|");
        System.out.println(" INPUTS                                 |");
        System.out.println("========================================|");
        System.out.println("                                        |");
        System.out.print(" >> Input Values ---------------------- | ");
        for (Neuron neuron : network.getLayers().get(0).getNeurons()) {
            System.out.print(neuron.getValue() + " ");
        }
        System.out.println("\n                                        |");
    }

    @Override
    public void showOutputs() {
        System.out.println("========================================|");
        System.out.println(" OUTPUTS                                |");
        System.out.println("========================================|");
        System.out.println("                                        |");
        System.out.print(" >> Output Values --------------------- | ");
        for (Neuron neuron : network.getLayers().get(network.getLayers().size() - 1).getNeurons()) {
            System.out.print(neuron.getValue() + " ");
        }
        System.out.println("\n                                        |");
    }

    /*
     * GETTERS AND SETTERS
     */

    public static Network getNetwork() { return network; }
}
