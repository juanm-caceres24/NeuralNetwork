package src.utils;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

import src.Setup;
import src.models.Network;

import java.util.List;

public class FileUtils {

    /*
     * ATTRIBUTES
     */

    private Network network;
    private String configFilePath;

    /*
     * CONSTRUCTORS
     */

    public FileUtils(Network network) {
        this.network = network;
        this.configFilePath = Setup.getConfigFilePath();
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

    public void saveNetworkIntoSetup() {
        // Save the network's weights and biases into the setup file
        Setup.setWeights(network.getWeights());
        Setup.setBiases(network.getBiases());
        saveSetupIntoFile();
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

    /*
     * SERIALIZATION HELPERS
     */
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
}
