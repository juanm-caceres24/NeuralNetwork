package src.userinterface;

public interface UserInterface  {

    /*
     * METHODS
     */

    public abstract void showError(Integer errorCode);
    public abstract void showNetwork();
    public abstract void showInputs();
    public abstract void showOutputs();
}
