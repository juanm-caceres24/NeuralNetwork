package src;

import src.models.Network;
import src.user_interface.UserInterface;
import src.user_interface.impl.Console;
import src.utils.Trainer;

public class Main {

    /*
     * ATTRIBUTES
     */

    private static UserInterface userInterface;
    private static Network network;
    private static Trainer trainer;

    /*
     * MAIN METHOD
     */
    public static void main(String args[]) {
        network = new Network();
        userInterface = new Console(network);
        trainer = new Trainer(network);
        userInterface.showNetwork();
        trainer.train();
        network.predict();
        userInterface.showNetwork();
        userInterface.showInputs();
        userInterface.showOutputs();
    }
}
