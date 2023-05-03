package src;
public class NeuronConnectionStruct {
    
    public Neuron neuron;
    public Double weight;

    public NeuronConnectionStruct(Neuron neuron, Double weight) {
        this.neuron = neuron;
        this.weight = weight;
    }
}
