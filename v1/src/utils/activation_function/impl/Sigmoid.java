package v1.src.utils.activation_function.impl;

import v1.src.utils.activation_function.ActivationFunction;

public class Sigmoid implements ActivationFunction {

    /*
     * CONSTRUCTORS
     */
    
    public Sigmoid() { }

    /*
     * METHODS
     */

    @Override
    public double activate(double x) {
        return 1 / (1 + Math.pow(Math.E, - x));
    }

    @Override
    public double derivative(double x) {
        double y = this.activate(x);
        return y * (1 - y);
    }
}
