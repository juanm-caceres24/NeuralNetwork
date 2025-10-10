package src.activation_function.impl;

import src.activation_function.ActivationFunction;

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
