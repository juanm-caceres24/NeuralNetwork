package src.utils.activation_function.impl;

import src.utils.activation_function.ActivationFunction;

public class ReLU implements ActivationFunction {

    /*
     * CONSTRUCTORS
     */

    public ReLU() { }

    /*
     * METHODS
     */

    @Override
    public Double activate(Double x) {
        return Math.max(0, x);
    }

    @Override
    public Double derivative(Double x) {
        return x > 0 ? 1.0 : 0.0;
    }
}
