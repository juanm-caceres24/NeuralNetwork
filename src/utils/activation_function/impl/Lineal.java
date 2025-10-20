package src.utils.activation_function.impl;

import src.utils.activation_function.ActivationFunction;

public class Lineal implements ActivationFunction {

    /*
     * CONSTRUCTORS
     */

    public Lineal() { }

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
