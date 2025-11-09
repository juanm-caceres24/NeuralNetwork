package v1.src.utils.activation_function.impl;

import v1.src.utils.activation_function.ActivationFunction;

public class ReLU implements ActivationFunction {

    /*
     * CONSTRUCTORS
     */

    public ReLU() { }

    /*
     * METHODS
     */

    @Override
    public double activate(double x) {
        return Math.max(0, x);
    }

    @Override
    public double derivative(double x) {
        return x > 0 ? 1.0 : 0.0;
    }
}
