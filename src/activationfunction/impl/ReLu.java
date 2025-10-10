package src.activationfunction.impl;

import src.activationfunction.ActivationFunction;

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
