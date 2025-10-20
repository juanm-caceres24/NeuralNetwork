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
        System.out.printf("========================================|\n");
        System.out.printf(" MODE SELECTION                         | '0'=Predict\n");
        System.out.printf("                                        | '1'=Create\n");
        System.out.printf("                                        | '2'=Load\n");
        System.out.printf("                                        | '3'=Train from File\n");
        System.out.printf("                                        | '4'=Train from Demo Test\n");
        System.out.printf("                                        | '5'=Exit (default=Predict)\n");
        System.out.printf("========================================|\n");
        System.out.printf("                                    >>> | Select mode: ");
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
            case "5":
                return 5;
            default:
                this.showError(0);
                return 0;
        }        
    }

    @Override
    public void showError(Integer errorCode) {
        System.out.printf("========================================|\n");
        System.out.printf(" ERROR                                  |\n");
        switch (errorCode) {
            case 0:
                System.out.printf(" Invalid input. Selecting default mode.\n");
                break;
            case 1:
                System.out.printf(" File not found.\n");
                break;
            case 2:
                System.out.printf(" Unable to save file.\n");
                break;
            default:
                System.out.printf(" Unknown error.\n");
                break;
        }
        System.out.printf("========================================|\n");
    }

    @Override
    public void showNetwork() {
        System.out.printf("========================================|\n");
        System.out.printf(" NETWORK                                |\n");
        System.out.printf("========================================|\n");
        for (Layer layer : network.getLayers()) {
            System.out.printf("                                        |\n");
            System.out.printf(" >> | Layer Id ------------------------ | %d\n", layer.getLayerId());
            if (layer.getNextLayer() == null) {
                System.out.printf("    | Next Layer Id ------------------- | (none)\n");
            } else {
                System.out.printf("    | Next Layer Id ------------------- | %d\n", layer.getNextLayer().getLayerId());
            }
            if (layer.getPreviousLayer() == null) {
                System.out.printf("    | Previous Layer Id --------------- | (none)\n");
            } else {
                System.out.printf("    | Previous Layer Id --------------- | %d\n", layer.getPreviousLayer().getLayerId());
            }
            for (Neuron neuron : layer.getNeurons()) {
                System.out.printf("    +                                   |\n");
                System.out.printf("    | ->> | Neuron Id ----------------- | %d\n", neuron.getNeuronId());
                System.out.printf("    |     | Value --------------------- | %.8f\n", neuron.getValue());
                System.out.printf("    |     | Bias ---------------------- | %.8f\n", neuron.getBias());
                System.out.printf("    |     | Activation Function ------- | %s\n", neuron.getActivationFunction().getClass().getSimpleName());
                if (neuron.getForwardWeights() != null) {
                    for (Double w : neuron.getForwardWeights()) {
                        if (w == neuron.getForwardWeights().get(0)) {
                            System.out.printf("    |     | Forward Weights ----------- | %.8f\n", w);
                        } else {
                            System.out.printf("    |     |                           + | %.8f\n", w);
                        }
                    }
                } else {
                    System.out.printf("    |     | Forward Weights ----------- | (none)\n");
                }
                if (neuron.getBackwardWeights() != null) {
                    for (Double w : neuron.getBackwardWeights()) {
                        if (w == neuron.getBackwardWeights().get(0)) {
                            System.out.printf("    |     | Backward Weights ---------- | %.8f\n", w);
                        } else {
                            System.out.printf("    |     |                           + | %.8f\n", w);
                        }
                    }
                } else {
                    System.out.printf("    |     | Backward Weights ---------- | (none)\n");
                }
            }
        }
        System.out.printf("                                        |\n");
    }

    @Override
    public void showInputs() {
        System.out.printf("========================================|\n");
        System.out.printf(" INPUTS                                 |\n");
        System.out.printf("========================================|\n");
        System.out.printf("                                        |\n");
        Integer inputLayerIndex = 0;
        for (Neuron neuron : network.getLayers().get(0).getNeurons()) {
            if (inputLayerIndex == 0) {
                System.out.printf(" >> Input Values ---------------------- | %.8f\n", neuron.getValue());
            } else {
                System.out.printf("                                      + | %.8f\n", neuron.getValue());
            }
            inputLayerIndex++;
        }
        System.out.printf("                                        |\n");
    }

    @Override
    public void showOutputs() {
        System.out.printf("========================================|\n");
        System.out.printf(" OUTPUTS                                |\n");
        System.out.printf("========================================|\n");
        System.out.printf("                                        |\n");
        Integer outputLayerIndex = 0;
        for (Neuron neuron : network.getLayers().get(network.getLayers().size() - 1).getNeurons()) {
            if (outputLayerIndex == 0) {
                System.out.printf(" >> Output Values --------------------- | %.8f\n", neuron.getValue());
            } else {
                System.out.printf("                                      + | %.8f\n", neuron.getValue());
            }
            outputLayerIndex++;
        }
        System.out.printf("                                        |\n");
    }

    /*
     * GETTERS AND SETTERS
     */

    public Network getNetwork() { return network; }
    public void setNetwork(Network network) { this.network = network; }
    public Scanner getScanner() { return scanner; }
}
