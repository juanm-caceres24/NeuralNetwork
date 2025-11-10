package v1.src.utils.trainer;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import v1.src.Setup;
import v1.src.models.Layer;
import v1.src.models.Network;
import v1.src.models.Neuron;
import v1.src.utils.loss.Loss;
import v1.src.utils.loss.impl.MSELoss;
import v1.src.utils.trainer.test_generator.TestGenerator;
import v1.src.utils.trainer.test_generator.impl.BinToHex;
//import src.utils.trainer.test_generator.impl.XOR;

public class Trainer {

    /*
     * ATTRIBUTES
     */

    // Network instance
    private Network network;

    // Error/Loss function
    private Loss loss;

    // Demo generator
    private TestGenerator demoGenerator;

    /*
     * CONSTRUCTORS
     */

    public Trainer(Network network) {
        this.network = network;
        this.loss = new MSELoss();
        this.demoGenerator = new BinToHex();
    }

    /*
     * METHODS
     */

    public void train() {
        double LEARNING_RATE = Setup.getLearningRate();
        int EPOCHS = Setup.getEpochs();
        int BATCH_SIZE = Setup.getBatchSize();
        double[][][] TRAINING_DATA = Setup.getTrainingData();
        // Simple training loop using stochastic gradient descent (SGD)
        List<Integer> indexes = new ArrayList<>();
        for (int i = 0; i < TRAINING_DATA.length; i++) {
            indexes.add(i);
        }
        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            // Shuffle samples each epoch to aid SGD convergence
            Collections.shuffle(indexes);
            // Process data in mini-batches
            for (int start = 0; start < indexes.size(); start += BATCH_SIZE) {
                int end = Math.min(start + BATCH_SIZE, indexes.size());
                int actualBatch = end - start;
                // Prepare accumulators for biases and weights per layer
                Layer[] layers = network.getLayers();
                int L = layers.length;
                if (L < 2) continue; // nothing to update
                double[][] biasGradSums = new double[L][]; // biasGradSums[layer][neuron]
                double[][][] weightGradSums = new double[L][][]; // weightGradSums[layer][to][from]
                // Initialize accumulator shapes
                for (int l = 1; l < L; l++) {
                    Layer layer = layers[l];
                    int neuronsCount = layer.getNeurons().length;
                    int prevNeurons = layer.getPreviousLayer() != null ? layer.getPreviousLayer().getNeurons().length : 0;
                    biasGradSums[l] = new double[neuronsCount];
                    weightGradSums[l] = new double[neuronsCount][prevNeurons];
                    // Initialize values to 0.0 to avoid NullPointerException when accumulating
                    for (int ni = 0; ni < neuronsCount; ni++) {
                        biasGradSums[l][ni] = 0.0;
                        for (int pi = 0; pi < prevNeurons; pi++) {
                            weightGradSums[l][ni][pi] = 0.0;
                        }
                    }
                }
                // Accumulate gradients for each sample in the batch
                for (int p = start; p < end; p++) {
                    int sample = indexes.get(p);
                    // Forward and backward to compute deltas
                    network.forward(TRAINING_DATA[sample][0]);
                    backward(TRAINING_DATA[sample][1]);
                    // Accumulate per-layer gradients (skip input layer l=0)
                    for (int l = 1; l < L; l++) {
                        Layer layer = layers[l];
                        Layer prev = layer.getPreviousLayer();
                        for (int j = 0; j < layer.getNeurons().length; j++) {
                            Neuron neuron = layer.getNeurons()[j];
                            double delta = neuron.getDelta();
                            biasGradSums[l][j] += delta;
                            if (neuron.getBackwardWeights() != null && prev != null) {
                                for (int k = 0; k < neuron.getBackwardWeights().length; k++) {
                                    double prevVal = prev.getNeurons()[k].getActivation();
                                    weightGradSums[l][j][k] += delta * prevVal;
                                }
                            }
                        }
                    }
                }
                // Apply averaged gradients to update biases and backward weights
                for (int l = 1; l < L; l++) {
                    Layer layer = layers[l];
                    for (int j = 0; j < layer.getNeurons().length; j++) {
                        Neuron neuron = layer.getNeurons()[j];
                        // Update bias: average over batch
                        double biasGradAvg = biasGradSums[l][j] / (double) actualBatch;
                        double newBias = neuron.getBias() - LEARNING_RATE * biasGradAvg;
                        neuron.setBias(newBias);
                        // Update backward (incoming) weights
                        if (neuron.getBackwardWeights() != null) {
                            for (int k = 0; k < neuron.getBackwardWeights().length; k++) {
                                double gradAvg = weightGradSums[l][j][k] / (double) actualBatch;
                                double updated = neuron.getBackwardWeights()[k] - LEARNING_RATE * gradAvg;
                                neuron.getBackwardWeights()[k] = updated;
                            }
                        }
                    }
                }
                // Synchronize backwardWeights to forwardWeights for consistency
                for (int l = 0; l < layers.length - 1; l++) {
                    Layer layer = layers[l];
                    Layer next = layers[l + 1];
                    for (int i = 0; i < layer.getNeurons().length; i++) {
                        Neuron neuron = layer.getNeurons()[i];
                        if (neuron.getForwardWeights() == null) continue;
                        for (int j = 0; j < neuron.getForwardWeights().length; j++) {
                            Neuron nextNeuron = next.getNeurons()[j];
                            if (nextNeuron.getBackwardWeights() != null) {
                                double w = nextNeuron.getBackwardWeights()[i];
                                neuron.getForwardWeights()[j] = w;
                            }
                        }
                    }
                }
            }
            // DEBUG
            System.out.format("DEBUG - Epoch: %d/%d\n", epoch, EPOCHS);
        }
    }

    private void backward(double[] y) {
        // Compute output layer deltas using loss derivative and activation derivative
        Layer[] layers = network.getLayers();
        int L = layers.length;
        if (L == 0) return;
        Layer outputLayer = layers[L - 1];
        // Gather predicted outputs
        double[] predicted = new double[outputLayer.getNeurons().length];
        for (int i = 0; i < predicted.length; i++) {
            predicted[i] = outputLayer.getNeurons()[i].getActivation();
        }
        double[] lossDeriv = loss.derivative(predicted, y);
        // Set deltas for output neurons
        for (int i = 0; i < outputLayer.getNeurons().length; i++) {
            Neuron neuron = outputLayer.getNeurons()[i];
            // derivative of activation evaluated at z
            double activationDeriv = neuron.getActivationFunction().derivative(neuron.getZ());
            neuron.setDelta(lossDeriv[i] * activationDeriv);
        }
        // Backpropagate deltas through hidden layers using Layer.backPropagate()
        for (int l = L - 2; l >= 1; l--) { // Skip input layer at index 0
            Layer layer = layers[l];
            layer.backPropagate();
        }
    }

    public void generateTestTrainingValues() {
        double[][][] TRAINING_DATA = this.demoGenerator.generateTrainingData();
        Setup.setTrainingData(TRAINING_DATA);
    }

    /*
     * GETTERS AND SETTERS
     */

    public Network getNetwork() { return network; }
    public Loss getLoss() { return loss; }
    public TestGenerator getDemoGenerator() { return demoGenerator; }
}
