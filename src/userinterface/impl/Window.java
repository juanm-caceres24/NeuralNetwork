package src.userinterface.impl;

import src.models.Network;
import src.userinterface.UserInterface;

public class Window implements UserInterface {

    /*
     * ATTRIBUTES
     */
    
    private static Network network;

    /*
     * CONSTRUCTORS
     */

    public Window(Network network) {
        Window.network = network;
    }

    /*
     * METHODS
     */

    @Override
    public void showError(Integer errorCode) { }

    @Override
    public void showNetwork() { }

    @Override
    public void showInputs() { }

    @Override
    public void showOutputs() { }

    /*
     * GETTERS AND SETTERS
     */

    public static Network getNetwork() { return network; }
}
