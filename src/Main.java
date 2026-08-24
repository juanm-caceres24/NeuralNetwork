package src;

import src.models.Network;
import src.models.Neuron;
import src.api.PredictionServer;
import src.user_interface.UserInterface;
import src.user_interface.impl.Console;
import src.utils.FileUtils;
import src.utils.trainer.Trainer;

public class Main {

    /*
     * ATTRIBUTES
     */

    // Objects instances
    private static UserInterface userInterface;
    private static FileUtils fileUtils;
    private static Network network;
    private static Trainer trainer;

    /*
     * MAIN METHOD
     */
    public static void main(String args[]) {
        boolean running = true;
        userInterface = new Console(null);
        while (running) {
            switch (userInterface.requestModeSelection()) {
                case 0:
                    predictNetwork();
                    break;
                case 1:
                    createNetwork();
                    break;
                case 2:
                    importNetwork();
                    break;
                case 3:
                    trainNetwork();
                    break;
                case 4:
                    testTraining();
                    break;
                case 5:
                    showNetwork();
                    break;
                case 6:
                    calculateErrorFromFile();
                    break;
                case 7:
                    startApiMode();
                    break;
                case 8:
                    running = false;
                    break;
                default:
                    userInterface.showError(null);
                    break;
            }
        }
    }

    /*
     * METHODS
     */

    public static void predictNetwork() {
        try {
            fileUtils.importInput();
            network.predict();
            fileUtils.exportOutput();
            userInterface.showInputs();
            userInterface.showOutputs();
        } catch (Exception e) {
            userInterface.showError(1);
            e.printStackTrace();
        }
    }

    public static void createNetwork() {
        try {
            fileUtils = new FileUtils(null);
            fileUtils.importSetup();
            Setup.initializeRandomNetwork();
            network = new Network();
            userInterface.setNetwork(network);
            fileUtils.setNetwork(network);
            fileUtils.exportNetwork();
        } catch (Exception e) {
            userInterface.showError(2);
            e.printStackTrace();
        }
    }

    public static void importNetwork() {
        try {
            fileUtils = new FileUtils(null);
            fileUtils.importSetup();
            network = new Network();
            userInterface.setNetwork(network);
            fileUtils.setNetwork(network);
        } catch (Exception e) {
            userInterface.showError(3);
            e.printStackTrace();
        }
    }

    public static void trainNetwork() {
        try {
            trainer = new Trainer(network);
            fileUtils.importTrainingData();
            trainer.train();
            network.saveNetwork();
            fileUtils.exportNetwork();
        } catch (Exception e) {
            userInterface.showError(4);
            e.printStackTrace();
        }
    }

    public static void testTraining() {
        try {
            trainer = new Trainer(network);
            trainer.generateTestTrainingValues();
            fileUtils.exportTrainingData();
            trainer.train();
            network.saveNetwork();
            fileUtils.exportNetwork();
        } catch (Exception e) {
            userInterface.showError(4);
            e.printStackTrace();
        }
    }

    public static void showNetwork() {
        try {
            userInterface.showNetwork();
        } catch (Exception e) {
            userInterface.showError(5);
            e.printStackTrace();
        }
    }

    public static void calculateErrorFromFile() {
        try {
            if (network == null) {
                userInterface.showError(3);
                return;
            }
            fileUtils = new FileUtils(network);
            fileUtils.importSetup();
            fileUtils.importTrainingData();
            double[][][] trainingData = Setup.getTrainingData();
            double totalAbsoluteError = 0.0;
            long comparisons = 0;
            for (double[][] dataPair : trainingData) {
                if (dataPair == null || dataPair.length < 2) continue;
                network.forward(dataPair[0]);
                double[] target = dataPair[1];
                Neuron[] outputNeurons = network.getLayers()[network.getLayers().length - 1].getNeurons();
                int outputCount = Math.min(outputNeurons.length, target.length);
                for (int i = 0; i < outputCount; i++) {
                    totalAbsoluteError += Math.abs(outputNeurons[i].getActivation() - target[i]);
                    comparisons++;
                }
            }
            double meanAbsoluteError = comparisons == 0 ? 0.0 : totalAbsoluteError / comparisons;
            userInterface.showFileError(totalAbsoluteError, meanAbsoluteError, trainingData.length);
        } catch (Exception e) {
            userInterface.showError(6);
            e.printStackTrace();
        }
    }

    public static void startApiMode() {
        try {
            if (network == null) {
                fileUtils = new FileUtils(null);
                fileUtils.importSetup();
                network = new Network();
            }
            userInterface.showAPIMode();
            new PredictionServer(network).startAndWait();
        } catch (Exception e) {
            userInterface.showError(7);
            e.printStackTrace();
        }
    }
}
