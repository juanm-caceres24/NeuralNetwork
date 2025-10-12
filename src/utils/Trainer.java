package src.utils;

import java.util.ArrayList;
import java.util.Collections;
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
    private Double[][][] trainingData;

    /*
     * CONSTRUCTORS
     */

    public Trainer(Network network) {
        this.network = network;
        this.learningRate = Setup.getLearningRate();
        this.epochs = Setup.getEpochs();
        this.batchSize = Setup.getBatchSize();
        this.loss = new MSELoss();
        this.trainingData = Setup.getTrainingData();
    }

    /*
     * METHODS
     */

    public void train() {
        // Simple training loop using stochastic gradient descent
        int computeEvery = Math.max(1, this.epochs / 10);
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < trainingData.length; i++) {
            indices.add(i);
        }
        for (int epoch = 0; epoch < this.epochs; epoch++) {
            // Shuffle samples each epoch to aid SGD convergence
            Collections.shuffle(indices);
            for (int idx = 0; idx < indices.size(); idx++) {
                int sample = indices.get(idx);
                // Set input values and forward
                network.forward(trainingData[sample][0]);
                // Compute backward deltas using target
                backward(trainingData[sample][1]);
                // Apply gradients to update weights and biases
                applyGradients();
            }
            // Compute loss periodically
            if ((epoch + 1) % computeEvery == 0 || epoch == this.epochs - 1) {
                ArrayList<Layer> layers = network.getLayers();
                Layer outputLayer = layers.get(layers.size() - 1);
                for (int s = 0; s < trainingData.length; s++) {
                    network.forward(trainingData[s][0]);
                    Double[] pred = new Double[outputLayer.getNeurons().size()];
                    for (int i = 0; i < pred.length; i++) {
                        pred[i] = outputLayer.getNeurons().get(i).getValue();
                    }
                }
            }
        }
    }

    private void backward(Double[] y) {
        // Compute output layer deltas using loss derivative and activation derivative
        ArrayList<Layer> layers = network.getLayers();
        int L = layers.size();
        if (L == 0) return;
        Layer outputLayer = layers.get(L - 1);
        // gather predicted outputs
        Double[] predicted = new Double[outputLayer.getNeurons().size()];
        for (int i = 0; i < predicted.length; i++) {
            predicted[i] = outputLayer.getNeurons().get(i).getValue();
        }
        Double[] lossDeriv = loss.derivative(predicted, y);
        // Set deltas for output neurons
        for (int i = 0; i < outputLayer.getNeurons().size(); i++) {
            Neuron neuron = outputLayer.getNeurons().get(i);
            // derivative of activation evaluated at z
            double activationDeriv = neuron.getActivationFunction().derivative(neuron.getZ());
            neuron.setDelta(lossDeriv[i] * activationDeriv);
        }
        // Backpropagate deltas through hidden layers using Layer.backPropagate()
        for (int l = L - 2; l >= 1; l--) { // skip input layer at index 0
            Layer layer = layers.get(l);
            layer.backPropagate();
        }
    }

    private void applyGradients() {
        // Update weights and biases using current deltas stored in neurons (SGD, per-sample)
        ArrayList<Layer> layers = network.getLayers();
        if (layers.size() < 2) return;
        // Update biases and incoming (backward) weights only. We'll sync these to forwardWeights after.
        for (int l = 1; l < layers.size(); l++) { // start from first hidden layer (skip input layer)
            Layer layer = layers.get(l);
            Layer prev = layer.getPreviousLayer();
            ArrayList<Neuron> neurons = layer.getNeurons();
            for (int i = 0; i < neurons.size(); i++) {
                Neuron neuron = neurons.get(i);
                // Update bias: b = b - lr * delta
                double newBias = neuron.getBias() - this.learningRate * neuron.getDelta();
                neuron.setBias(newBias);
                // Update backward (incoming) weights: w = w - lr * delta * prevValue
                if (neuron.getBackwardWeights() != null && prev != null) {
                    for (int k = 0; k < neuron.getBackwardWeights().size(); k++) {
                        double prevValue = prev.getNeurons().get(k).getValue();
                        double grad = neuron.getDelta() * prevValue; // dLoss/dw = delta * input
                        double updated = neuron.getBackwardWeights().get(k) - this.learningRate * grad;
                        neuron.getBackwardWeights().set(k, updated);
                    }
                }
                // Do NOT update forwardWeights here to avoid double updates/overwrites.
            }
        }
        // Synchronize backwardWeights (canonical) to forwardWeights so both views match.
        for (int l = 0; l < layers.size() - 1; l++) {
            Layer layer = layers.get(l);
            Layer next = layers.get(l + 1);
            for (int i = 0; i < layer.getNeurons().size(); i++) {
                Neuron neuron = layer.getNeurons().get(i);
                if (neuron.getForwardWeights() == null) continue;
                for (int j = 0; j < neuron.getForwardWeights().size(); j++) {
                    // forward weight from neuron i (in layer) to neuron j (in next)
                    // is stored as next.getNeurons().get(j).backwardWeights.get(i)
                    Neuron nextNeuron = next.getNeurons().get(j);
                    if (nextNeuron.getBackwardWeights() != null) {
                        Double w = nextNeuron.getBackwardWeights().get(i);
                        neuron.getForwardWeights().set(j, w);
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
    public Double[][][] getTrainingData() { return trainingData; }
}
