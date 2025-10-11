package src.utils;

public class MSELoss implements Loss {
    
    @Override
    public Double loss(
            Double[] predicted,
            Double[] target) {
                
        double sum = 0.0;
        for (int i = 0; i < predicted.length; i++) {
            double p = predicted[i] != null ? predicted[i] : 0.0;
            double t = target[i] != null ? target[i] : 0.0;
            double d = p - t;
            sum += d * d;
        }
        return sum / predicted.length;
    }

    @Override
    public Double[] derivative(
            Double[] predicted,
            Double[] target) {

        Double[] out = new Double[predicted.length];
        for (int i = 0; i < predicted.length; i++) {
            double p = predicted[i] != null ? predicted[i] : 0.0;
            double t = target[i] != null ? target[i] : 0.0;
            // d/dp (1/N * sum (p - t)^2) = 2*(p - t)/N
            out[i] = 2.0 * (p - t) / predicted.length;
        }
        return out;
    }
}
