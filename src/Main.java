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
                    trainNetwork();
                    break;
                case 4:
                    testTraining();
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
        try {
            fileUtils.importInput();
            network.predict();
            fileUtils.exportOutput();
            userInterface.showInputs();
            userInterface.showOutputs();
        } catch (Exception e) {
            userInterface.showError(1);
            //e.printStackTrace();
        }
    }

    public static void createNetwork() {
        try {
            fileUtils = new FileUtils(null);
            fileUtils.importSetup();
            network = new Network();
            userInterface.setNetwork(network);
            fileUtils.setNetwork(network);
            fileUtils.exportNetwork();
            userInterface.showNetwork();
        } catch (Exception e) {
            userInterface.showError(2);
            //e.printStackTrace();
        }
    }

    public static void loadNetwork() {
        try {
            fileUtils = new FileUtils(null);
            fileUtils.importSetup();
            network = new Network();
            userInterface.setNetwork(network);
            fileUtils.setNetwork(network);
            userInterface.showNetwork();
        } catch (Exception e) {
            userInterface.showError(3);
            //e.printStackTrace();
        }
    }

    public static void trainNetwork() {
        try {
            trainer = new Trainer(network);
            fileUtils.importTrainingData();
            trainer.train();
            network.saveNetwork();
            fileUtils.exportNetwork();
            userInterface.showNetwork();
        } catch (Exception e) {
            userInterface.showError(4);
            //e.printStackTrace();
        }
    }

    public static void testTraining() {
        try {
            trainer = new Trainer(network);
            trainer.generateTestTrainingValues();
            fileUtils.exportTrainingData();
            trainer.train();
            network.saveNetwork();
            fileUtils.exportNetwork();
            userInterface.showNetwork();
        } catch (Exception e) {
            userInterface.showError(4);
            //e.printStackTrace();
        }
    }
}
