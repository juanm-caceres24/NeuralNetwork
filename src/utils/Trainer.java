package src.utils;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import src.Setup;
import src.models.Layer;
import src.models.Network;
import src.models.Neuron;
import src.utils.loss.Loss;
import src.utils.loss.impl.MSELoss;

public class Trainer {

    /*
     * ATTRIBUTES
     */

    // Network instance
    private Network network;

    // Error/Loss function
    private Loss loss;

    // Setup parameters
    private Double learningRate;
    private Integer epochs;
    private Integer batchSize;
    private Double[][][] trainingData;

    /*
     * CONSTRUCTORS
     */

    public Trainer(Network network) {
        this.network = network;
        this.loss = new MSELoss();
        this.learningRate = Setup.getLearningRate();
        this.epochs = Setup.getEpochs();
        this.batchSize = Setup.getBatchSize();
        this.trainingData = Setup.getTrainingData();
    }

    /*
     * METHODS
     */

    public void train() {
        // Simple training loop using stochastic gradient descent (SGD)
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < trainingData.length; i++) {
            indices.add(i);
        }
        for (int epoch = 0; epoch < this.epochs; epoch++) {
            // Shuffle samples each epoch to aid SGD convergence
            Collections.shuffle(indices);
            // Process data in mini-batches
            for (int start = 0; start < indices.size(); start += this.batchSize) {
                int end = Math.min(start + this.batchSize, indices.size());
                int actualBatch = end - start;
                // Prepare accumulators for biases and weights per layer
                ArrayList<Layer> layers = network.getLayers();
                int L = layers.size();
                if (L < 2) continue; // nothing to update
                double[][] biasGradSums = new double[L][]; // biasGradSums[layer][neuron]
                double[][][] weightGradSums = new double[L][][]; // weightGradSums[layer][to][from]
                // Initialize accumulator shapes
                for (int l = 1; l < L; l++) {
                    Layer layer = layers.get(l);
                    int neuronsCount = layer.getNeurons().size();
                    int prevNeurons = layer.getPreviousLayer() != null ? layer.getPreviousLayer().getNeurons().size() : 0;
                    biasGradSums[l] = new double[neuronsCount];
                    weightGradSums[l] = new double[neuronsCount][prevNeurons];
                }
                // Accumulate gradients for each sample in the batch
                for (int p = start; p < end; p++) {
                    int sample = indices.get(p);
                    // Forward and backward to compute deltas
                    network.forward(trainingData[sample][0]);
                    backward(trainingData[sample][1]);
                    // Accumulate per-layer gradients (skip input layer l=0)
                    for (int l = 1; l < L; l++) {
                        Layer layer = layers.get(l);
                        Layer prev = layer.getPreviousLayer();
                        for (int j = 0; j < layer.getNeurons().size(); j++) {
                            Neuron neuron = layer.getNeurons().get(j);
                            double delta = neuron.getDelta();
                            biasGradSums[l][j] += delta;
                            if (neuron.getBackwardWeights() != null && prev != null) {
                                for (int k = 0; k < neuron.getBackwardWeights().size(); k++) {
                                    double prevVal = prev.getNeurons().get(k).getValue();
                                    weightGradSums[l][j][k] += delta * prevVal;
                                }
                            }
                        }
                    }
                }
                // Apply averaged gradients to update biases and backward weights
                for (int l = 1; l < L; l++) {
                    Layer layer = layers.get(l);
                    for (int j = 0; j < layer.getNeurons().size(); j++) {
                        Neuron neuron = layer.getNeurons().get(j);
                        // Update bias: average over batch
                        double biasGradAvg = biasGradSums[l][j] / (double) actualBatch;
                        double newBias = neuron.getBias() - this.learningRate * biasGradAvg;
                        neuron.setBias(newBias);
                        // Update backward (incoming) weights
                        if (neuron.getBackwardWeights() != null) {
                            for (int k = 0; k < neuron.getBackwardWeights().size(); k++) {
                                double gradAvg = weightGradSums[l][j][k] / (double) actualBatch;
                                double updated = neuron.getBackwardWeights().get(k) - this.learningRate * gradAvg;
                                neuron.getBackwardWeights().set(k, updated);
                            }
                        }
                    }
                }
                // Synchronize backwardWeights to forwardWeights for consistency
                for (int l = 0; l < layers.size() - 1; l++) {
                    Layer layer = layers.get(l);
                    Layer next = layers.get(l + 1);
                    for (int i = 0; i < layer.getNeurons().size(); i++) {
                        Neuron neuron = layer.getNeurons().get(i);
                        if (neuron.getForwardWeights() == null) continue;
                        for (int j = 0; j < neuron.getForwardWeights().size(); j++) {
                            Neuron nextNeuron = next.getNeurons().get(j);
                            if (nextNeuron.getBackwardWeights() != null) {
                                Double w = nextNeuron.getBackwardWeights().get(i);
                                neuron.getForwardWeights().set(j, w);
                            }
                        }
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
        // Gather predicted outputs
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
        for (int l = L - 2; l >= 1; l--) { // Skip input layer at index 0
            Layer layer = layers.get(l);
            layer.backPropagate();
        }
    }

    /*
     * GETTERS AND SETTERS
     */

    public Network getNetwork() { return network; }
    public Loss getLoss() { return loss; }
    public Double getLearningRate() { return learningRate; }
    public Integer getEpochs() { return epochs; }
    public Integer getBatchSize() { return batchSize; }
    public Double[][][] getTrainingData() { return trainingData; }
}
