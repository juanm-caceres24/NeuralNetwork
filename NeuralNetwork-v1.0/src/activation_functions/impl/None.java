package src.activation_functions.impl;

import src.activation_functions.ActivationFunction;

public class None implements ActivationFunction {

    @Override
    public Double applyActivationFunction(Double value) {
        return value;
    }
}
