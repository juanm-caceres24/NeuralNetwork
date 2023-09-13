package src.activation_function.impl;

import src.activation_function.ActivationFunction;

public class Sigmoid implements ActivationFunction {

    @Override
    public Double applyActivationFunction(Double value) {
        return 1 / (1 + Math.pow(Math.E, - value));
    }
}
