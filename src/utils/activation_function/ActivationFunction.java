package src.utils.activation_function;

public interface ActivationFunction {

    /*
     * METHODS
     */

    public abstract double activate(double value);
    public abstract double derivative(double value);
}
