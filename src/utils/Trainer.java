package src.utils;

import java.util.ArrayList;
import java.util.List;

import src.Setup;
import src.models.Layer;
import src.models.Network;
import src.models.Neuron;

public class Trainer {

    /*
     * ATTRIBUTES
     */

    private Network network;

    // Setup parameters
    private Double learningRate;
    private Integer epochs;
    private Integer batchSize;
    private Loss loss;
    private Double[][] trainingInputs;
    private Double[][] trainingOutputs;

    /*
     * CONSTRUCTORS
     */

    public Trainer(Network network) {
        this.network = network;
        this.learningRate = Setup.getLearningRate();
        this.epochs = Setup.getEpochs();
        this.batchSize = Setup.getBatchSize();
        this.loss = new MSELoss();
        this.trainingInputs = Setup.getTrainingInputs();
        this.trainingOutputs = Setup.getTrainingOutputs();
    }

    /*
     * METHODS
     */

    public void train() {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < trainingInputs.length; i += batchSize) {
                Double[][] XBatch = new Double[Math.min(batchSize, trainingInputs.length - i)][];
                Double[][] YBatch = new Double[Math.min(batchSize, trainingOutputs.length - i)][];
                for (int j = 0; j < XBatch.length; j++) {
                    XBatch[j] = trainingInputs[i + j];
                    YBatch[j] = trainingOutputs[i + j];
                }
                for (int j = 0; j < XBatch.length; j++) {
                    network.forward(XBatch[j]);
                    this.backward(YBatch[j]);
                    this.applyGradients(learningRate, (double) XBatch.length);
                }
            }
        }
    }

    private void backward(Double[] y) {
        List<Layer> layers = network.getLayers();
        Layer outputLayer = layers.get(layers.size() - 1);
        Double[] predicted = new Double[outputLayer.getNeurons().size()];
        for (int i = 0; i < predicted.length; i++) {
            predicted[i] = outputLayer.getNeurons().get(i).getValue();
        }
        Double[] dLoss = loss.derivative(predicted, y);
        for (int j = 0; j < outputLayer.getNeurons().size(); j++) {
            Neuron neuron = outputLayer.getNeurons().get(j);
            Double dz = neuron.getActivationFunction().derivative(neuron.getZ());
            neuron.setDelta(dLoss[j] * dz);
        }
        for (int l = layers.size() - 2; l > 0; l--) {
            Layer layer = layers.get(l);
            Layer nextLayer = layer.getNextLayer();
            for (int j = 0; j < layer.getNeurons().size(); j++) {
                Neuron neuron = layer.getNeurons().get(j);
                double sum = 0.0;
                ArrayList<Double> fw = neuron.getForwardWeights();
                if (fw != null) {
                    for (int k = 0; k < fw.size(); k++) {
                        Double w = fw.get(k);
                        Neuron nextNeuron = nextLayer.getNeurons().get(k);
                        sum += (w != null ? w : 0.0) * nextNeuron.getDelta();
                    }
                }
                Double dz = neuron.getActivationFunction().derivative(neuron.getZ());
                neuron.setDelta(sum * dz);
            }
        }
    }

    private void applyGradients(
            Double lr,
            Double batchSize) {

        List<Layer> layers = network.getLayers();
        for (int l = 1; l < layers.size(); l++) {
            Layer layer = layers.get(l);
            Layer prev = layer.getPreviousLayer();
            for (int j = 0; j < layer.getNeurons().size(); j++) {
                Neuron neuron = layer.getNeurons().get(j);
                Double delta = neuron.getDelta();
                neuron.setBias(neuron.getBias() - (lr * delta / batchSize));
                ArrayList<Double> bw = neuron.getBackwardWeights();
                if (bw != null) {
                    for (int i = 0; i < bw.size(); i++) {
                        Double prevOut = prev.getNeurons().get(i).getValue();
                        Double grad = delta * prevOut; // dL/dw = delta_j * output_i
                        bw.set(i, bw.get(i) - (lr * grad / batchSize));
                    }
                    neuron.setBackwardWeights(bw);
                }
                if (prev != null) {
                    for (int i = 0; i < prev.getNeurons().size(); i++) {
                        Neuron prevNeuron = prev.getNeurons().get(i);
                        ArrayList<Double> prevFw = prevNeuron.getForwardWeights();
                        if (prevFw != null && j < prevFw.size()) {
                            if (neuron.getBackwardWeights() != null && i < neuron.getBackwardWeights().size()) {
                                prevFw.set(j, neuron.getBackwardWeights().get(i));
                                prevNeuron.setForwardWeights(prevFw);
                            }
                        }
                    }
                }
            }
        }
    }

    /*
     * GETTERS AND SETTERS
     */

    public Network getNetwork() { return network; }
    public Double getLearningRate() { return learningRate; }
    public Integer getEpochs() { return epochs; }
    public Integer getBatchSize() { return batchSize; }
    public Loss getLoss() { return loss; }
    public Double[][] getTrainingInputs() { return trainingInputs; }
    public Double[][] getTrainingOutputs() { return trainingOutputs; }
}
