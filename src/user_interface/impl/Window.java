package src.user_interface.impl;

import src.models.Network;
import src.user_interface.UserInterface;

public class Window implements UserInterface {

    /*
     * ATTRIBUTES
     */
    
    // Network instance
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
