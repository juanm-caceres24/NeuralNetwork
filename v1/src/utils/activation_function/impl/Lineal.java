package v1.src.utils.activation_function.impl;

import v1.src.utils.activation_function.ActivationFunction;

public class Lineal implements ActivationFunction {

    /*
     * CONSTRUCTORS
     */

    public Lineal() { }

    /*
     * METHODS
     */

    @Override
    public double activate(double x) {
        return x;
    }

    @Override
    public double derivative(double x) {
        return 1.0;
    }
}
