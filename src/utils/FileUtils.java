package src.utils;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

import src.Setup;
import src.models.Layer;
import src.models.Network;
import src.models.Neuron;

import java.util.ArrayList;
import java.util.List;

public class FileUtils {

    /*
     * ATTRIBUTES
     */

    private Network network;
    private String configFilePath;
    private String trainingDataFilePath;
    private String inputValuesFilePath;
    private String outputValuesFilePath;

    /*
     * CONSTRUCTORS
     */

    public FileUtils(Network network) {
        this.network = network;
        this.configFilePath = Setup.getConfigFilePath();
        this.trainingDataFilePath = Setup.getTrainingDataFilePath();
        this.inputValuesFilePath = Setup.getInputValuesFilePath();
        this.outputValuesFilePath = Setup.getOutputValuesFilePath();
    }

    /*
     * METHODS
     */

    public void dumpLineIntoFile(
            String line,
            String filePath) {
        
        // Implementation for writing 'line' into the file at 'filePath'
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(
                filePath,
                true))) {
            writer.write(line);
            writer.newLine();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void saveSetupIntoFile() {
        // First clear the file
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(configFilePath))) {
            writer.write("");
        } catch (IOException e) {
            e.printStackTrace();
        }
        // Then write the configuration parameters in a parseable bracket format
        dumpLineIntoFile("BIASES=" + toString(Setup.getBiases()), configFilePath);
        dumpLineIntoFile("WEIGHTS=" + toString(Setup.getWeights()), configFilePath);
        dumpLineIntoFile("ACTIVATION_FUNCTIONS=" + toString(Setup.getActivationFunctions()), configFilePath);
        dumpLineIntoFile("LEARNING_RATE=" + Setup.getLearningRate(), configFilePath);
        dumpLineIntoFile("EPOCHS=" + Setup.getEpochs(), configFilePath);
        dumpLineIntoFile("BATCH_SIZE=" + Setup.getBatchSize(), configFilePath);
    }

    public void saveNetworkIntoFile() {
        // Save the network's weights and biases into the setup file
        Setup.setWeights(network.getWeights());
        Setup.setBiases(network.getBiases());
        Setup.setActivationFunctions(network.getActivationFunctions());
        saveSetupIntoFile();
    }

    public void readInputValuesFromFile() {
        // Reads the input values from the input file and sets them into Setup
        // Expected format: one numeric value per line. Empty lines are ignored.
        try (java.io.BufferedReader reader = new java.io.BufferedReader(new java.io.FileReader(inputValuesFilePath))) {
            java.util.List<Double> values = new java.util.ArrayList<>();
            String line;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue; // skip empty lines
                // Parse the whole line as a double (one value per line)
                values.add(Double.parseDouble(line));
            }
            if (!values.isEmpty()) {
                Double[] inputValues = values.toArray(new Double[0]);
                Setup.setInputValues(inputValues);
            }
        } catch (IOException | NumberFormatException e) {
            e.printStackTrace();
        }
    }

    public void saveOutputValuesIntoFile() {
        // Write each output value on its own line (overwrites file)
        if (this.outputValuesFilePath == null) return;
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputValuesFilePath))) {
            if (this.network == null) return;
            ArrayList<Layer> layers = this.network.getLayers();
            if (layers == null || layers.isEmpty()) return;
            Layer lastLayer = layers.get(layers.size() - 1);
            if (lastLayer == null || lastLayer.getNeurons() == null) return;

            for (Neuron neuron : lastLayer.getNeurons()) {
                Double v = neuron.getValue();
                writer.write(v == null ? "" : v.toString());
                writer.newLine();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void loadSetupFromFile() {
        // Reads the configuration file and sets up the parameters accordingly
        try (java.io.BufferedReader reader = new java.io.BufferedReader(new java.io.FileReader(configFilePath))) {
            String line;
            while ((line = reader.readLine()) != null) {
                String[] parts = line.split("=");
                if (parts.length != 2) continue;
                String key = parts[0];
                String value = parts[1].trim();
                switch (key) {
                    case "BIASES":
                        Setup.setBiases(parseDoubleMatrix(value));
                        break;
                    case "WEIGHTS":
                        Setup.setWeights(parseDoubleTensor(value));
                        break;
                    case "ACTIVATION_FUNCTIONS":
                        Setup.setActivationFunctions(parseIntegerArray(value));
                        break;
                    case "LEARNING_RATE":
                        Setup.setLearningRate(Double.parseDouble(value));
                        break;
                    case "EPOCHS":
                        Setup.setEpochs(Integer.parseInt(value));
                        break;
                    case "BATCH_SIZE":
                        Setup.setBatchSize(Integer.parseInt(value));
                        break;
                    default:
                        // Unknown key
                        break;
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private Double[] parseDoubleArray(String value) {
        if (value == null) return new Double[0];
        String s = value.trim();
        if (s.startsWith("[") && s.endsWith("]")) s = s.substring(1, s.length() - 1);
        if (s.trim().isEmpty()) return new Double[0];
        String[] parts = s.split(",");
        Double[] array = new Double[parts.length];
        for (int i = 0; i < parts.length; i++) {
            array[i] = Double.parseDouble(parts[i].trim());
        }
        return array;
    }

    private Double[][] parseDoubleMatrix(String value) {
        if (value == null) return new Double[0][];
        String s = value.trim();
        if (!s.startsWith("[") || !s.endsWith("]")) return new Double[0][];
        // remove outer brackets
        s = s.substring(1, s.length() - 1).trim();
        List<String> rows = new java.util.ArrayList<>();
        int level = 0;
        StringBuilder cur = new StringBuilder();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '[') {
                level++;
                if (level == 1) continue; // skip the bracket that starts the row
            }
            if (c == ']') {
                level--;
                if (level == 0) {
                    rows.add(cur.toString());
                    cur.setLength(0);
                    continue;
                }
            }
            if (level >= 1) cur.append(c);
            // ignore commas at level 0
        }
        Double[][] matrix = new Double[rows.size()][];
        for (int i = 0; i < rows.size(); i++) {
            String row = rows.get(i).trim();
            // split row by commas
            if (row.isEmpty()) {
                matrix[i] = new Double[0];
                continue;
            }
            String[] parts = row.split(",");
            matrix[i] = new Double[parts.length];
            for (int j = 0; j < parts.length; j++) {
                matrix[i][j] = Double.parseDouble(parts[j].trim());
            }
        }
        return matrix;
    }

    private Double[][][] parseDoubleTensor(String value) {
        if (value == null) return new Double[0][][];
        String s = value.trim();
        if (!s.startsWith("[") || !s.endsWith("]")) return new Double[0][][];
        // remove outer brackets
        s = s.substring(1, s.length() - 1).trim();
        List<String> mats = new java.util.ArrayList<>();
        int level = 0;
        StringBuilder cur = new StringBuilder();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '[') {
                level++;
            }
            if (level > 0) cur.append(c);
            if (c == ']') {
                level--;
                if (level == 0) {
                    mats.add(cur.toString());
                    cur.setLength(0);
                }
            }
        }
        Double[][][] tensor = new Double[mats.size()][][];
        for (int i = 0; i < mats.size(); i++) {
            tensor[i] = parseDoubleMatrix(mats.get(i));
        }
        return tensor;
    }

    private Integer[] parseIntegerArray(String value) {
        if (value == null) return new Integer[0];
        String s = value.trim();
        if (s.startsWith("[") && s.endsWith("]")) s = s.substring(1, s.length() - 1);
        if (s.trim().isEmpty()) return new Integer[0];
        String[] parts = s.split(",");
        Integer[] array = new Integer[parts.length];
        for (int i = 0; i < parts.length; i++) {
            array[i] = Integer.parseInt(parts[i].trim());
        }
        return array;
    }

    private String toString(Double[] arr) {
        if (arr == null) return "[]";
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i = 0; i < arr.length; i++) {
            if (i > 0) sb.append(",");
            sb.append(arr[i]);
        }
        sb.append("]");
        return sb.toString();
    }

    private String toString(Double[][] mat) {
        if (mat == null) return "[]";
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i = 0; i < mat.length; i++) {
            if (i > 0) sb.append(",");
            sb.append(toString(mat[i]));
        }
        sb.append("]");
        return sb.toString();
    }

    private String toString(Double[][][] tensor) {
        if (tensor == null) return "[]";
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i = 0; i < tensor.length; i++) {
            if (i > 0) sb.append(",");
            sb.append(toString(tensor[i]));
        }
        sb.append("]");
        return sb.toString();
    }

    private String toString(Integer[] arr) {
        if (arr == null) return "[]";
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i = 0; i < arr.length; i++) {
            if (i > 0) sb.append(",");
            sb.append(arr[i]);
        }
        sb.append("]");
        return sb.toString();
    }

    /*
     * GETTERS & SETTERS
     */

    public Network getNetwork() { return network; }
    public void setNetwork(Network network) { this.network = network; }
    public String getConfigFilePath() { return configFilePath; }
    public String getTrainingDataFilePath() { return trainingDataFilePath; }
    public String getInputValuesFilePath() { return inputValuesFilePath; }
    public String getOutputValuesFilePath() { return outputValuesFilePath; }
}
