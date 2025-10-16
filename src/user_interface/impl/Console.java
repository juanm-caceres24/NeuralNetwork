package src.user_interface.impl;

import java.util.Scanner;

import src.models.Layer;
import src.models.Network;
import src.models.Neuron;
import src.user_interface.UserInterface;

public class Console implements UserInterface {

    /*
     * ATTRIBUTES
     */

    // Object instances
    private Network network;
    private Scanner scanner;

    /*
     * CONSTRUCTORS
     */

    public Console(Network network) {
        this.network = network;
        scanner = new Scanner(System.in);
    }

    /*
     * METHODS
     */

    @Override
    public Integer requestModeSelection() {
        System.out.println("========================================|");
        System.out.println(" MODE SELECTION                         | '0'=Predict, '1'=Train, '2'=Create, '3'=Load, '4'=Exit (default=Predict)");
        System.out.println("========================================|");
        System.out.print("                                    >>> | Select mode: ");
        String input = scanner.nextLine();
        switch (input) {
            case "0":
                return 0;
            case "1":
                return 1;
            case "2":
                return 2;
            case "3":
                return 3;
            case "4":
                return 4;
            default:
                this.showError(0);
                return 0;
        }        
    }

    @Override
    public void showError(Integer errorCode) {
        System.out.println("========================================|");
        System.out.print(" ERROR                                  |");
        switch (errorCode) {
            case 0:
                System.out.println(" Invalid input. Selecting default mode.");
                break;
            case 1:
                System.out.println(" File not found.");
                break;
            case 2:
                System.out.println(" Unable to save file.");
                break;
            default:
                System.out.println(" Unknown error.");
                break;
        }
        System.out.println("========================================|");
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

    public Network getNetwork() { return network; }
    public void setNetwork(Network network) { this.network = network; }
    public Scanner getScanner() { return scanner; }
}
