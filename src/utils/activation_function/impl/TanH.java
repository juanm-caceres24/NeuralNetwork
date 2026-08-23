package src.utils.activation_function.impl;

import src.utils.activation_function.ActivationFunction;

public class TanH implements ActivationFunction {

    /*
     * CONSTRUCTORS
     */

    public TanH() { }

    /*
     * METHODS
     */

    @Override
    public double activate(double x) {
        return Math.tanh(x);
    }

    @Override
    public double derivative(double x) {
        double y = this.activate(x);
        return 1 - Math.pow(y, 2);
    }
}
