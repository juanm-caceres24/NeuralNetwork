package src;

public class NeuronConnection {
    
    public final Neuron neuron;
    public final Double weight;

    public NeuronConnection(Neuron neuron, Double weight) {
        this.neuron = neuron;
        this.weight = weight;
    }
}
