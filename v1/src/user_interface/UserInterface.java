package v1.src.user_interface;

import v1.src.models.Network;

public interface UserInterface  {

    /*
     * METHODS
     */

    public abstract Integer requestModeSelection();
    public abstract void showError(Integer errorCode);
    public abstract void showNetwork();
    public abstract void showInputs();
    public abstract void showOutputs();
    public abstract void setNetwork(Network network);
}
