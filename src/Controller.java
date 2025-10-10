package src;

import src.models.Layer;
import src.models.Network;
import src.models.Neuron;
import src.userinterface.UserInterface;
import src.userinterface.impl.Console;

public class Controller {

    /*
     * ATTRIBUTES
     */

    private static UserInterface userInterface;
    private static Network network;

    /*
     * CONSTRUCTORS
     */

    public Controller() {
        network = new Network();
        userInterface = new Console(network);
    }

    /*
     * METHODS
     */

    public void predict() {
        // Load input values into the input layer
        for (Neuron neuron : network.getLayers().get(0).getNeurons()) {
            neuron.setValue(Setup.getInputValues()[neuron.getNeuronId()]);
        }
        // Propagate values through the network
        for (Layer layer : network.getLayers()) {
            if (layer.getPreviousLayer() != null) {
                layer.feedForward();
            }
        }        
    }
    
    /*
     * GETTERS AND SETTERS
     */

    public UserInterface getUserInterface() { return userInterface; }

    public Network getNetwork() { return network; }
}
