package src;

public class Main {

    /*
     * ATTRIBUTES
     */

    private static Controller controller;

    /*
     * MAIN METHOD
     */
    public static void main(String args[]) {
        controller = new Controller();
        controller.predict();
        controller.getUserInterface().showNetwork();
        controller.getUserInterface().showInputs();
        controller.getUserInterface().showOutputs();
    }
}
