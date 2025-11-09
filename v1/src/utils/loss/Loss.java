package v1.src.utils.loss;

public interface Loss {

    public abstract double loss(double[] predicted, double[] target);
    public abstract double[] derivative(double[] predicted, double[] target);
}
