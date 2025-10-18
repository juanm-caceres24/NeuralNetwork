package src.utils.activation_function.impl;

import src.utils.activation_function.ActivationFunction;

public class None implements ActivationFunction {

    /*
     * CONSTRUCTORS
     */

    public None() { }

    /*
     * METHODS
     */

    @Override
    public Double activate(Double x) {
        return x;
    }

    @Override
    public Double derivative(Double x) {
        return 1.0;
    }
}
