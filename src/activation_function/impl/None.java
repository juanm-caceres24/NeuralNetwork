package src.activation_function.impl;

import src.activation_function.ActivationFunction;

public class None implements ActivationFunction {

    /*
     * CONSTRUCTORS
     */

    public None() { }

    /*
     * METHODS
     */

    @Override
    public Double activate(Double value) {
        return value;
    }

    @Override
    public Double derivative(Double value) {
        return 1.0;
    }
}
