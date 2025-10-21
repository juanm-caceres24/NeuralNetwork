package src.utils.trainer;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import src.Setup;
import src.models.Layer;
import src.models.Network;
import src.models.Neuron;
import src.utils.loss.Loss;
import src.utils.loss.impl.MSELoss;
import src.utils.trainer.test_generator.TestGenerator;
import src.utils.trainer.test_generator.impl.BinToHex;
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
        Double LEARNING_RATE = Setup.getLearningRate();
        Integer EPOCHS = Setup.getEpochs();
        Integer BATCH_SIZE = Setup.getBatchSize();
        Double[][][] TRAINING_DATA = Setup.getTrainingData();
        // Simple training loop using stochastic gradient descent (SGD)
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < TRAINING_DATA.length; i++) {
            indices.add(i);
        }
        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            // Shuffle samples each epoch to aid SGD convergence
            Collections.shuffle(indices);
            // Process data in mini-batches
            for (int start = 0; start < indices.size(); start += BATCH_SIZE) {
                Integer end = Math.min(start + BATCH_SIZE, indices.size());
                Integer actualBatch = end - start;
                // Prepare accumulators for biases and weights per layer
                ArrayList<Layer> layers = network.getLayers();
                Integer L = layers.size();
                if (L < 2) continue; // nothing to update
                Double[][] biasGradSums = new Double[L][]; // biasGradSums[layer][neuron]
                Double[][][] weightGradSums = new Double[L][][]; // weightGradSums[layer][to][from]
                // Initialize accumulator shapes
                for (int l = 1; l < L; l++) {
                    Layer layer = layers.get(l);
                    Integer neuronsCount = layer.getNeurons().size();
                    Integer prevNeurons = layer.getPreviousLayer() != null ? layer.getPreviousLayer().getNeurons().size() : 0;
                    biasGradSums[l] = new Double[neuronsCount];
                    weightGradSums[l] = new Double[neuronsCount][prevNeurons];
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
                    Integer sample = indices.get(p);
                    // Forward and backward to compute deltas
                    network.forward(TRAINING_DATA[sample][0]);
                    backward(TRAINING_DATA[sample][1]);
                    // Accumulate per-layer gradients (skip input layer l=0)
                    for (int l = 1; l < L; l++) {
                        Layer layer = layers.get(l);
                        Layer prev = layer.getPreviousLayer();
                        for (int j = 0; j < layer.getNeurons().size(); j++) {
                            Neuron neuron = layer.getNeurons().get(j);
                            Double delta = neuron.getDelta();
                            biasGradSums[l][j] += delta;
                            if (neuron.getBackwardWeights() != null && prev != null) {
                                for (int k = 0; k < neuron.getBackwardWeights().size(); k++) {
                                    Double prevVal = prev.getNeurons().get(k).getActivation();
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
                        Double biasGradAvg = biasGradSums[l][j] / (double) actualBatch;
                        Double newBias = neuron.getBias() - LEARNING_RATE * biasGradAvg;
                        neuron.setBias(newBias);
                        // Update backward (incoming) weights
                        if (neuron.getBackwardWeights() != null) {
                            for (int k = 0; k < neuron.getBackwardWeights().size(); k++) {
                                Double gradAvg = weightGradSums[l][j][k] / (double) actualBatch;
                                Double updated = neuron.getBackwardWeights().get(k) - LEARNING_RATE * gradAvg;
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
        Integer L = layers.size();
        if (L == 0) return;
        Layer outputLayer = layers.get(L - 1);
        // Gather predicted outputs
        Double[] predicted = new Double[outputLayer.getNeurons().size()];
        for (int i = 0; i < predicted.length; i++) {
            predicted[i] = outputLayer.getNeurons().get(i).getActivation();
        }
        Double[] lossDeriv = loss.derivative(predicted, y);
        // Set deltas for output neurons
        for (int i = 0; i < outputLayer.getNeurons().size(); i++) {
            Neuron neuron = outputLayer.getNeurons().get(i);
            // derivative of activation evaluated at z
            Double activationDeriv = neuron.getActivationFunction().derivative(neuron.getZ());
            neuron.setDelta(lossDeriv[i] * activationDeriv);
        }
        // Backpropagate deltas through hidden layers using Layer.backPropagate()
        for (int l = L - 2; l >= 1; l--) { // Skip input layer at index 0
            Layer layer = layers.get(l);
            layer.backPropagate();
        }
    }

    public void generateDemoTestTrainingValues() {
        Double[][][] TRAINING_DATA = this.demoGenerator.generateTrainingData();
        Setup.setTrainingData(TRAINING_DATA);
    }

    /*
     * GETTERS AND SETTERS
     */

    public Network getNetwork() { return network; }
    public Loss getLoss() { return loss; }
    public TestGenerator getDemoGenerator() { return demoGenerator; }
}
