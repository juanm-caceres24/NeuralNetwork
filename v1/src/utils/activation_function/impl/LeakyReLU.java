package v1.src.utils.activation_function.impl;

import v1.src.utils.activation_function.ActivationFunction;

public class LeakyReLU implements ActivationFunction {

    /*
     * CONSTRUCTORS
     */

    public LeakyReLU() { }

    /*
     * METHODS
     */

    @Override
    public Double activate(Double x) {
        return (x > 0) ? x : 0.01 * x;
    }

    @Override
    public Double derivative(Double x) {
        return (x > 0) ? 1.0 : 0.01;
    }
}
