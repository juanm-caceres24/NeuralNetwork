package src.activation_functions.impl;

import src.activation_functions.ActivationFunction;

public class Sigmoid implements ActivationFunction {

    @Override
    public Double applyActivationFunction(Double value) {
        return 1 / (1 + Math.pow(Math.E, - value));
    }
}
