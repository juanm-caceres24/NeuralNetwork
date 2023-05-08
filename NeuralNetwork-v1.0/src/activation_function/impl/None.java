package src.activation_function.impl;

import src.activation_function.ActivationFunction;

public class None implements ActivationFunction {

    @Override
    public Double applyActivationFunction(Double value) {
        return value;
    }
}
