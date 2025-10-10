package src.activation_function.impl;

import src.activation_function.ActivationFunction;

public class TanH implements ActivationFunction {

    /*
     * CONSTRUCTORS
     */

    public TanH() { }

    /*
     * METHODS
     */

    @Override
    public Double activate(Double value) {
        return Math.tanh(value);
    }
}
