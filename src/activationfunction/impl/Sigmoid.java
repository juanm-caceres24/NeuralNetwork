package src.activationfunction.impl;

import src.activationfunction.ActivationFunction;

public class Sigmoid implements ActivationFunction {

    /*
     * CONSTRUCTORS
     */
    
    public Sigmoid() { }

    /*
     * METHODS
     */

    @Override
    public Double activate(Double value) {
        return 1 / (1 + Math.pow(Math.E, - value));
    }
}
