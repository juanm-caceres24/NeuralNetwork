package src.activation_function;

public interface ActivationFunction {

    /*
     * METHODS
     */
    
    public abstract Double activate(Double value);
    public abstract Double derivative(Double value);
}
