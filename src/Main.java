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
        //Main.initializeAndTrainNetwork();
        Main.loadAndPredictNetwork();
    }

    /*
     * METHODS
     */

    public static void initializeAndTrainNetwork() {
        Setup.generateTrainingOutput();
        network = new Network();
        userInterface = new Console(network);
        fileUtils = new FileUtils(network);
        trainer = new Trainer(network);
        userInterface.showNetwork();
        trainer.train();
        userInterface.showNetwork();
        network.saveNetwork();
        fileUtils.saveNetworkIntoSetup();
        fileUtils.saveSetupIntoFile();
    }

    public static void loadAndPredictNetwork() {
        fileUtils = new FileUtils(null);
        fileUtils.loadSetupFromFile();
        network = new Network();
        fileUtils.setNetwork(network);
        userInterface = new Console(network);
        userInterface.showNetwork();
        network.predict();
        userInterface.showInputs();
        userInterface.showOutputs();
    }
}
