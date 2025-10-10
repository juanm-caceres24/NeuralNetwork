package src.activationfunction.impl;

import src.activationfunction.ActivationFunction;

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
}
