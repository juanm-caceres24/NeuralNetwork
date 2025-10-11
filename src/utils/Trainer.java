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

    private static Network network;
    private static Loss loss;
    private static Double learningRate;

    /*
     * CONSTRUCTORS
     */

    public Trainer(Network network) {
        Trainer.network = network;
        Trainer.loss = new MSELoss();
        Trainer.learningRate = Setup.getLearningRate();
    }

    /*
     * METHODS
     */

    public void train() {
        for (int epoch = 0; epoch < Setup.getEpochs(); epoch++) {
            for (int i = 0; i < Setup.getTrainingInputs().length; i += Setup.getBatchSize()) {
                Double[][] XBatch = new Double[Math.min(Setup.getBatchSize(), Setup.getTrainingInputs().length - i)][];
                Double[][] YBatch = new Double[Math.min(Setup.getBatchSize(), Setup.getTrainingOutputs().length - i)][];
                for (int j = 0; j < XBatch.length; j++) {
                    XBatch[j] = Setup.getTrainingInputs()[i + j];
                    YBatch[j] = Setup.getTrainingOutputs()[i + j];
                }
                for (int j = 0; j < XBatch.length; j++) {
                    this.forward(XBatch[j]);
                    this.backward(YBatch[j]);
                    this.applyGradients(learningRate, (double) XBatch.length);
                }
            }
        }
    }

    private void forward(Double[] x) {
        // Set input layer values
        Layer inputLayer = network.getLayers().get(0);
        for (int i = 0; i < inputLayer.getNeurons().size(); i++) {
            inputLayer.getNeurons().get(i).setValue(x[i]);
        }
        // Feedforward through the network
        for (int i = 1; i < network.getLayers().size(); i++) {
            network.getLayers().get(i).feedForward();
        }
    }

    private void backward(Double[] y) {
        List<Layer> layers = network.getLayers();
        Layer outputLayer = layers.get(layers.size() - 1);

        // 1) obtener predicciones actuales
        Double[] predicted = new Double[outputLayer.getNeurons().size()];
        for (int i = 0; i < predicted.length; i++) {
            predicted[i] = outputLayer.getNeurons().get(i).getValue();
        }

        // 2) dL/dy usando la Loss
        Double[] dLoss = loss.derivative(predicted, y);

        // 3) delta salida = dL/dy * activation'(z)
        for (int j = 0; j < outputLayer.getNeurons().size(); j++) {
            Neuron neuron = outputLayer.getNeurons().get(j);
            Double dz = neuron.getActivationFunction().derivative(neuron.getZ());
            neuron.setDelta(dLoss[j] * dz);
        }

        // 4) backpropagate a capas ocultas (desde la última oculta hacia la primera)
        for (int l = layers.size() - 2; l > 0; l--) { // salta capa de entrada (índice 0)
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
        // actualiza desde la primera capa con backwardWeights (índice 1) hasta la capa de salida
        for (int l = 1; l < layers.size(); l++) {
            Layer layer = layers.get(l);
            Layer prev = layer.getPreviousLayer();
            for (int j = 0; j < layer.getNeurons().size(); j++) {
                Neuron neuron = layer.getNeurons().get(j);
                Double delta = neuron.getDelta();

                // bias
                neuron.setBias(neuron.getBias() - (lr * delta / batchSize));

                // backwardWeights: w_i_j (desde prev neurona i hacia esta neurona j)
                ArrayList<Double> bw = neuron.getBackwardWeights();
                if (bw != null) {
                    for (int i = 0; i < bw.size(); i++) {
                        Double prevOut = prev.getNeurons().get(i).getValue();
                        Double grad = delta * prevOut; // dL/dw = delta_j * output_i
                        bw.set(i, bw.get(i) - (lr * grad / batchSize));
                    }
                    neuron.setBackwardWeights(bw);
                }

                // sincronizar forwardWeights de la capa previa (prevNeurons.forwardWeights[j] == neuron.backwardWeights[i])
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
    public void setNetwork(Network network) { Trainer.network = network; }
    public Loss getLoss() { return loss; }
    public void setLoss(Loss loss) { Trainer.loss = loss; }
    public Double getLearningRate() { return learningRate; }
    public void setLearningRate(Double learningRate) { Trainer.learningRate = learningRate; }
}
