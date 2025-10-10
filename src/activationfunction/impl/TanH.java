package src.activationfunction.impl;

import src.activationfunction.ActivationFunction;

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
