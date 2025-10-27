package v1.src.utils.loss;

public interface Loss {

    public abstract Double loss(Double[] predicted, Double[] target);
    public abstract Double[] derivative(Double[] predicted, Double[] target);
}
