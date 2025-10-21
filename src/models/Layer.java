package src.models;

import java.util.ArrayList;

public class Layer{
    
    /*
     * ATTRIBUTES
     */

    // Layer identifier
    private Integer layerId;

    // Layer topology
    private ArrayList<Neuron> neurons;
    private Layer nextLayer;
    private Layer previousLayer;

    /*
     * CONSTRUCTORS
     */

    public Layer(Integer layerId, ArrayList<Neuron> neurons, Layer nextLayer, Layer previousLayer) {
        this.layerId = layerId;
        this.neurons = neurons;
        this.nextLayer = nextLayer;
        this.previousLayer = previousLayer;
    }

    /*
     * METHODS
     */

    public void feedForward() {
        ArrayList<Double> inputs = new ArrayList<>();
        for (Neuron neuron : this.previousLayer.getNeurons()) {
            inputs.add(neuron.getActivation());
        }
        for (Neuron neuron : this.neurons) {
            neuron.calculateForward(inputs);
        }
        for (Neuron neuron : this.neurons) {
            neuron.calculateActivation();
        }
    }

    public void backPropagate() {
        ArrayList<Double> inputs = new ArrayList<>();
        for (Neuron neuron : this.nextLayer.getNeurons()) {
            inputs.add(neuron.getDelta());
        }
        for (Neuron neuron : this.neurons) {
            neuron.calculateBackward(inputs);
        }
        for (Neuron neuron : this.neurons) {
            neuron.calculateDelta();
        }
    }

    /*
     * GETTERS AND SETTERS
     */

    public Integer getLayerId() { return layerId; }
    public ArrayList<Neuron> getNeurons() { return neurons; }
    public Layer getNextLayer() { return nextLayer; }
    public void setNextLayer(Layer nextLayer) { this.nextLayer = nextLayer; }
    public Layer getPreviousLayer() { return previousLayer; }
}
