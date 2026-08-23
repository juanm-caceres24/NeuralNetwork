package src.user_interface;

import src.models.Network;

public interface UserInterface  {

    /*
     * METHODS
     */

    public abstract Integer requestModeSelection();
    public abstract void showError(Integer errorCode);
    public abstract void showNetwork();
    public abstract void showInputs();
    public abstract void showOutputs();
    public abstract void showFileError(double totalAbsoluteError, double meanAbsoluteError, int sampleCount);
    public abstract void setNetwork(Network network);
}
