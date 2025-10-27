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
    public Double activate(Double x) {
        return 1 / (1 + Math.pow(Math.E, - x));
    }

    @Override
    public Double derivative(Double x) {
        Double y = this.activate(x);
        return y * (1 - y);
    }
}
