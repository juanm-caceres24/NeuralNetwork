package v1.src.utils.loss.impl;

import v1.src.utils.loss.Loss;

public class MSELoss implements Loss {
    
    @Override
    public double loss(double[] predicted, double[] target) {
        double sum = 0.0;
        for (int i = 0; i < predicted.length; i++) {
            double diff = predicted[i] - target[i];
            sum += diff * diff;
        }
        return sum / predicted.length;
    }

    @Override
    public double[] derivative(double[] predicted, double[] target) {
        double[] deriv = new double[predicted.length];
        int n = predicted.length;
        for (int i = 0; i < n; i++) {
            // Use sum-of-squares derivative (not averaged) to provide stronger gradients for small networks
            deriv[i] = 2.0 * (predicted[i] - target[i]);
        }
        return deriv;
    }
}
