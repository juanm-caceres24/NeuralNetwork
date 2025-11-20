package v1.src.models;

public class Layer{
    
    /*
     * ATTRIBUTES
     */

    // Layer identifier
    private Integer layerId;

    // Layer topology
    private Neuron[] neurons;
    private Layer nextLayer;
    private Layer previousLayer;

    /*
     * CONSTRUCTORS
     */

    public Layer(Integer layerId, Neuron[] neurons, Layer nextLayer, Layer previousLayer) {
        this.layerId = layerId;
        this.neurons = neurons;
        this.nextLayer = nextLayer;
        this.previousLayer = previousLayer;
    }

    /*
     * METHODS
     */

    public void feedForward() {
        double[] inputs = new double[this.previousLayer.getNeurons().length];
        for (int i = 0; i < this.previousLayer.getNeurons().length; i++) {
            inputs[i] = this.previousLayer.getNeurons()[i].getActivation();
        }
        for (Neuron neuron : this.neurons) {
            neuron.calculateForward(inputs);
            neuron.calculateActivation();
        }
    }

    public void backPropagate() {
        double[] inputs = new double[this.nextLayer.getNeurons().length];
        for (int i = 0; i < this.nextLayer.getNeurons().length; i++) {
            inputs[i] = this.nextLayer.getNeurons()[i].getDelta();
        }
        for (Neuron neuron : this.neurons) {
            neuron.calculateBackward(inputs);
            neuron.calculateDelta();
        }
    }

    /*
     * GETTERS AND SETTERS
     */

    public Integer getLayerId() { return layerId; }
    public Neuron[] getNeurons() { return neurons; }
    public Layer getNextLayer() { return nextLayer; }
    public void setNextLayer(Layer nextLayer) { this.nextLayer = nextLayer; }
    public Layer getPreviousLayer() { return previousLayer; }
}
