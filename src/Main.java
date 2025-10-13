package src;

import src.models.Network;
import src.user_interface.UserInterface;
import src.user_interface.impl.Console;
import src.utils.FileUtils;
import src.utils.Trainer;

public class Main {

    /*
     * ATTRIBUTES
     */

    private static UserInterface userInterface;
    private static FileUtils fileUtils;
    private static Network network;
    private static Trainer trainer;

    /*
     * MAIN METHOD
     */
    public static void main(String args[]) {
        Main.createNetwork();
        //Main.loadNetwork();
        Main.trainNetwork();
        Main.predictNetwork();
    }

    /*
     * METHODS
     */

    public static void createNetwork() {
        Setup.initializeFromLayerSizes();
        network = new Network();
        userInterface = new Console(network);
        fileUtils = new FileUtils(network);
        fileUtils.exportNetworkToFile();
        userInterface.showNetwork();
    }

    public static void loadNetwork() {
        fileUtils = new FileUtils(null);
        fileUtils.importSetupFromFile();
        Setup.initializeFromWeightsAndBiases();
        network = new Network();
        fileUtils.setNetwork(network);
        userInterface = new Console(network);
        userInterface.showNetwork();
    }

    public static void trainNetwork() {
        Setup.generateTestTrainingValues();
        trainer = new Trainer(network);
        trainer.train();
        network.saveNetwork();
        fileUtils.exportNetworkToFile();
        userInterface.showNetwork();
    }

    public static void predictNetwork() {
        fileUtils.importInputFromFile();
        network.setInputValues(Setup.getInputValues());
        network.predict();
        fileUtils.exportOutputToFile();
        userInterface.showInputs();
        userInterface.showOutputs();
    }
}
