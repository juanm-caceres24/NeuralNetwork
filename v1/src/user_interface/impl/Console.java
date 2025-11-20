package v1.src.user_interface.impl;

import java.util.Scanner;

import v1.src.models.Layer;
import v1.src.models.Network;
import v1.src.models.Neuron;
import v1.src.user_interface.UserInterface;

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
        System.out.printf(" MODE SELECTION                         |\n");
        System.out.printf("========================================|\n");
        System.out.printf("                                        |\n");
        System.out.printf("                            Predict='0' |\n");
        System.out.printf("                             Create='1' |\n");
        System.out.printf("                             Import='2' |\n");
        System.out.printf("                              Train='3' |\n");
        System.out.printf("                               Test='4' |\n");
        System.out.printf("                               Show='5' |\n");
        System.out.printf("                               Exit='6' |\n");
        System.out.printf("                      (default=Predict) |\n");
        System.out.printf("                                        |\n");
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
        System.out.printf(" ERROR                                  |");
        switch (errorCode) {
            case 0:
                System.out.printf(" Invalid input. Selecting default mode.\n");
                break;
            case 1:
                System.out.printf(" Problem in prediction.\n");
                break;
            case 2:
                System.out.printf(" Problem in creating network.\n");
                break;
            case 3:
                System.out.printf(" Problem in loading network.\n");
                break;
            case 4:
                System.out.printf(" Problem in training network.\n");
                break;
            case 5: 
                System.out.printf(" Problem showing network.\n");
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
                System.out.printf("    |     | Activation ---------------- | %.8f\n", neuron.getActivation());
                System.out.printf("    |     | Bias ---------------------- | %.8f\n", neuron.getBias());
                System.out.printf("    |     | Activation Function ------- | %s\n", neuron.getActivationFunction().getClass().getSimpleName());
                if (neuron.getForwardWeights() != null) {
                    for (int i = 0; i < neuron.getForwardWeights().length; i++) {
                        double w = neuron.getForwardWeights()[i];
                        if (w == neuron.getForwardWeights()[0]) {
                            System.out.printf("    |     | Forward Weights ----------- | (0) %.8f\n", w);
                        } else {
                            System.out.printf("    |     |                           + | (%d) %.8f\n", i, w);
                        }
                    }
                } else {
                    System.out.printf("    |     | Forward Weights ----------- | (none)\n");
                }
                if (neuron.getBackwardWeights() != null) {
                    for (int i = 0; i < neuron.getBackwardWeights().length; i++) {
                        double w = neuron.getBackwardWeights()[i];
                        if (w == neuron.getBackwardWeights()[0]) {
                            System.out.printf("    |     | Backward Weights ---------- | (0) %.8f\n", w);
                        } else {
                            System.out.printf("    |     |                           + | (%d) %.8f\n", i, w);
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
        int inputLayerIndex = 0;
        for (Neuron neuron : network.getLayers()[0].getNeurons()) {
            if (inputLayerIndex == 0) {
                System.out.printf(" >> Input Values ---------------------- | (0) %.8f\n", neuron.getActivation());
            } else {
                System.out.printf("                                      + | (%d) %.8f\n", inputLayerIndex, neuron.getActivation());
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
        int outputLayerIndex = 0;
        for (Neuron neuron : network.getLayers()[network.getLayers().length - 1].getNeurons()) {
            if (outputLayerIndex == 0) {
                System.out.printf(" >> Output Values --------------------- | (0) %.8f\n", neuron.getActivation());
            } else {
                System.out.printf("                                      + | (%d) %.8f\n", outputLayerIndex, neuron.getActivation());
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
