package src;

import src.models.Network;
import src.user_interface.UserInterface;
import src.user_interface.impl.Console;
import src.utils.FileUtils;
import src.utils.trainer.Trainer;

public class Main {

    /*
     * ATTRIBUTES
     */

    // Objects instances
    private static UserInterface userInterface;
    private static FileUtils fileUtils;
    private static Network network;
    private static Trainer trainer;

    /*
     * MAIN METHOD
     */
    public static void main(String args[]) {
        Boolean running = true;
        userInterface = new Console(null);
        while (running) {
            switch (userInterface.requestModeSelection()) {
                case 0:
                    predictNetwork();
                    break;
                case 1:
                    createNetwork();
                    break;
                case 2:
                    loadNetwork();
                    break;
                case 3:
                    trainNetworkFromFile();
                    break;
                case 4:
                    trainNetworkFromDemoTest();
                    break;
                case 5:
                    running = false;
                    break;
                default:
                    userInterface.showError(null);
                    break;
            }
        }
    }

    /*
     * METHODS
     */

    public static void predictNetwork() {
        fileUtils.importInput();
        network.predict();
        fileUtils.exportOutput();
        userInterface.showInputs();
        userInterface.showOutputs();
    }

    public static void createNetwork() {
        Setup.initializeRandomNetwork();
        network = new Network();
        userInterface.setNetwork(network);
        fileUtils = new FileUtils(network);
        fileUtils.exportNetwork();
        userInterface.showNetwork();
    }

    public static void loadNetwork() {
        fileUtils = new FileUtils(null);
        fileUtils.importSetup();
        network = new Network();
        userInterface.setNetwork(network);
        fileUtils.setNetwork(network);
        userInterface = new Console(network);
        userInterface.showNetwork();
    }

    public static void trainNetworkFromFile() {
        trainer = new Trainer(network);
        fileUtils.importTrainingData();
        trainer.train();
        network.saveNetwork();
        fileUtils.exportNetwork();
        userInterface.showNetwork();
    }

    public static void trainNetworkFromDemoTest() {
        trainer = new Trainer(network);
        trainer.generateDemoTestTrainingValues();
        fileUtils.exportTrainingData();
        trainer.train();
        network.saveNetwork();
        fileUtils.exportNetwork();
        userInterface.showNetwork();
    }
}
