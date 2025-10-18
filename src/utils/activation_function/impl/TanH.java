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
    public Double activate(Double x) {
        return Math.tanh(x);
    }

    @Override
    public Double derivative(Double x) {
        Double y = this.activate(x);
        return 1 - Math.pow(y, 2);
    }
}
