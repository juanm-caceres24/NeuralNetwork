package src.activation_function.impl;

import src.activation_function.ActivationFunction;

public class ReLu implements ActivationFunction {

    /*
     * CONSTRUCTORS
     */

    public ReLu() { }

    /*
     * METHODS
     */

    @Override
    public Double activate(Double value) {
        return Math.max(0, value);
    }
}
