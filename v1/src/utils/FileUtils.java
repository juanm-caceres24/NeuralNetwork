package v1.src.utils;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import v1.src.Setup;
import v1.src.models.Layer;
import v1.src.models.Network;
import v1.src.models.Neuron;

public class FileUtils {

    /*
     * ATTRIBUTES
     */

    // Network instance
    private Network network;

    /*
     * CONSTRUCTORS
     */

    public FileUtils(Network network) {
        this.network = network;
    }

    /*
     * METHODS
     */

    public void dumpLine(String line, String filePath) {
        // Implementation for writing 'line' into the file at 'filePath'
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath, true))) {
            writer.write(line);
            writer.newLine();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void exportSetupToFile() {
        String CONFIG_FILE_PATH = Setup.getConfigFilePath();
        // First clear the file
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(CONFIG_FILE_PATH))) {
            writer.write("");
        } catch (IOException e) {
            e.printStackTrace();
        }
        // Then write the configuration parameters in a parseable bracket format
        dumpLine("LAYERS_SIZE=" + toString(Setup.getLayersSize()), CONFIG_FILE_PATH);
        dumpLine("BIASES=" + toString(Setup.getBiases()), CONFIG_FILE_PATH);
        dumpLine("WEIGHTS=" + toString(Setup.getWeights()), CONFIG_FILE_PATH);
        dumpLine("ACTIVATION_FUNCTIONS=" + toString(Setup.getActivationFunctions()), CONFIG_FILE_PATH);
        dumpLine("LEARNING_RATE=" + Setup.getLearningRate(), CONFIG_FILE_PATH);
        dumpLine("EPOCHS=" + Setup.getEpochs(), CONFIG_FILE_PATH);
        dumpLine("BATCH_SIZE=" + Setup.getBatchSize(), CONFIG_FILE_PATH);
        dumpLine("TEST_TRAINING_DATA_LENGTH=" + Setup.getTestTrainingDataLength(), CONFIG_FILE_PATH);
    }

    public void importSetup() {
        String CONFIG_FILE_PATH = Setup.getConfigFilePath();
        // Reads the configuration file and sets up the parameters accordingly
        try (java.io.BufferedReader reader = new java.io.BufferedReader(new java.io.FileReader(CONFIG_FILE_PATH))) {
            String line;
            while ((line = reader.readLine()) != null) {
                String[] parts = line.split("=");
                if (parts.length != 2) continue;
                String key = parts[0];
                String value = parts[1].trim();
                switch (key) {
                    case "LAYERS_SIZE":
                        Setup.setLayersSize(parseIntegerArray(value));
                        break;
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
                    case "TEST_TRAINING_DATA_LENGTH":
                        Setup.setTestTrainingDataLength(Integer.parseInt(value));
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

    public void exportNetwork() {
        // Save the network's weights and biases into the setup file
        exportSetupToFile();
    }

    public void importInput() {
        String INPUT_VALUES_FILE_PATH = Setup.getInputValuesFilePath();
        // Reads the input values from the input file and sets them into Setup
        try (BufferedReader reader = new BufferedReader(new FileReader(INPUT_VALUES_FILE_PATH))) {
            List<Double> values = new ArrayList<>();
            String line;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue; // skip empty lines
                // Parse the whole line as a double (one value per line)
                values.add(Double.parseDouble(line));
            }
            if (!values.isEmpty()) {
                double[] inputValues = new double[values.size()];
                for (int i = 0; i < values.size(); i++) {
                    inputValues[i] = values.get(i);
                }
                Setup.setInputValues(inputValues);
            }
        } catch (IOException | NumberFormatException e) {
            e.printStackTrace();
        }
    }

    public void exportOutput() {
        String OUTPUT_VALUES_FILE_PATH = Setup.getOutputValuesFilePath();
        // Write each output value on its own line (overwrites file)
        if (OUTPUT_VALUES_FILE_PATH == null) return;
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(OUTPUT_VALUES_FILE_PATH))) {
            if (this.network == null) return;
            Layer[] layers = this.network.getLayers();
            if (layers == null || layers.length == 0) return;
            Layer lastLayer = layers[layers.length - 1];
            if (lastLayer == null || lastLayer.getNeurons() == null) return;
            for (Neuron neuron : lastLayer.getNeurons()) {
                Double v = neuron.getActivation();
                writer.write(v == null ? "" : v.toString());
                writer.newLine();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void importTrainingData() {
        String TRAINING_DATA_FILE_PATH = Setup.getTrainingDataFilePath();
        // Reads the training data from the training data file and sets it into Setup
        List<double[][]> dataList = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(TRAINING_DATA_FILE_PATH))) {
            String line;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue; // skip empty lines
                // Support format: input_csv|target_csv  (both sides are comma-separated doubles)
                // Backwards-compatible: if no '|' is present, treat whole line as input and leave target empty
                String inputPart = line;
                String targetPart = "";
                int sep = line.indexOf('|');
                if (sep >= 0) {
                    inputPart = line.substring(0, sep).trim();
                    targetPart = line.substring(sep + 1).trim();
                }
                double[] inputRow = new double[0];
                double[] targetRow = new double[0];
                if (!inputPart.isEmpty()) {
                    String[] parts = inputPart.split(",");
                    inputRow = new double[parts.length];
                    for (int i = 0; i < parts.length; i++) {
                        String tok = parts[i].trim();
                        if (tok.isEmpty() || tok.equalsIgnoreCase("null")) {
                            inputRow[i] = 0.0; // fallback for missing values
                        } else {
                            try {
                                inputRow[i] = Double.parseDouble(tok);
                            } catch (NumberFormatException ex) {
                                // fallback to 0.0 on parse error
                                inputRow[i] = 0.0;
                            }
                        }
                    }
                }
                if (!targetPart.isEmpty()) {
                    String[] parts = targetPart.split(",");
                    targetRow = new double[parts.length];
                    for (int i = 0; i < parts.length; i++) {
                        String tok = parts[i].trim();
                        if (tok.isEmpty() || tok.equalsIgnoreCase("null")) {
                            targetRow[i] = 0.0;
                        } else {
                            try {
                                targetRow[i] = Double.parseDouble(tok);
                            } catch (NumberFormatException ex) {
                                targetRow[i] = 0.0;
                            }
                        }
                    }
                }
                dataList.add(new double[][] { inputRow, targetRow });
            }
            // Convert List<double[][]> to double[][][] (samples x 2 x N)
            double[][][] trainingData = new double[dataList.size()][][];
            for (int i = 0; i < dataList.size(); i++) {
                trainingData[i] = dataList.get(i);
            }
            // If no records, set an empty training data array to avoid NPEs downstream
            if (trainingData.length == 0) {
                Setup.setTrainingData(new double[0][][]);
            } else {
                Setup.setTrainingData(trainingData);
            }
        } catch (IOException | NumberFormatException e) {
            e.printStackTrace();
            // on error, ensure training data is at least an empty array to avoid NPE downstream
            Setup.setTrainingData(new double[0][][]);
        }
    }

    public void exportTrainingData() {
        String TRAINING_DATA_FILE_PATH = Setup.getTrainingDataFilePath();
        // Writes the training data from Setup into the training data file
        double[][][] trainingData = Setup.getTrainingData();
        if (trainingData == null) return;
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(TRAINING_DATA_FILE_PATH))) {
            for (double[][] dataPair : trainingData) {
                if (dataPair == null || dataPair.length == 0) continue;
                double[] inputData = dataPair.length > 0 ? dataPair[0] : new double[0];
                double[] targetData = dataPair.length > 1 ? dataPair[1] : new double[0];
                StringBuilder line = new StringBuilder();
                line.append(joinCSV(inputData));
                // write separator and target CSV (empty target allowed)
                line.append("|");
                line.append(joinCSV(targetData));
                writer.write(line.toString());
                writer.newLine();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private String joinCSV(double[] array) {
        if (array == null || array.length == 0) return "";
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < array.length; i++) {
            if (i > 0) sb.append(",");
            sb.append(array[i]);
        }
        return sb.toString();
    }

    @SuppressWarnings("unused")
    private double[] parseDoubleArray(String value) {
        if (value == null) return new double[0];
        String s = value.trim();
        if (s.startsWith("[") && s.endsWith("]")) s = s.substring(1, s.length() - 1);
        if (s.trim().isEmpty()) return new double[0];
        String[] parts = s.split(",");
        double[] array = new double[parts.length];
        for (int i = 0; i < parts.length; i++) {
            array[i] = Double.parseDouble(parts[i].trim());
        }
        return array;
    }

    private double[][] parseDoubleMatrix(String value) {
        if (value == null) return new double[0][];
        String s = value.trim();
        if (!s.startsWith("[") || !s.endsWith("]")) return new double[0][];
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
        double[][] matrix = new double[rows.size()][];
        for (int i = 0; i < rows.size(); i++) {
            String row = rows.get(i).trim();
            // split row by commas
            if (row.isEmpty()) {
                matrix[i] = new double[0];
                continue;
            }
            String[] parts = row.split(",");
            matrix[i] = new double[parts.length];
            for (int j = 0; j < parts.length; j++) {
                matrix[i][j] = Double.parseDouble(parts[j].trim());
            }
        }
        return matrix;
    }

    private double[][][] parseDoubleTensor(String value) {
        if (value == null) return new double[0][][];
        String s = value.trim();
        if (!s.startsWith("[") || !s.endsWith("]")) return new double[0][][];
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
        double[][][] tensor = new double[mats.size()][][];
        for (int i = 0; i < mats.size(); i++) {
            tensor[i] = parseDoubleMatrix(mats.get(i));
        }
        return tensor;
    }

    private int[] parseIntegerArray(String value) {
        if (value == null) return new int[0];
        String s = value.trim();
        if (s.startsWith("[") && s.endsWith("]")) s = s.substring(1, s.length() - 1);
        if (s.trim().isEmpty()) return new int[0];
        String[] parts = s.split(",");
        int[] array = new int[parts.length];
        for (int i = 0; i < parts.length; i++) {
            array[i] = Integer.parseInt(parts[i].trim());
        }
        return array;
    }

    private String toString(double[] array) {
        if (array == null) return "[]";
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i = 0; i < array.length; i++) {
            if (i > 0) sb.append(",");
            sb.append(array[i]);
        }
        sb.append("]");
        return sb.toString();
    }

    private String toString(double[][] matrix) {
        if (matrix == null) return "[]";
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i = 0; i < matrix.length; i++) {
            if (i > 0) sb.append(",");
            sb.append(toString(matrix[i]));
        }
        sb.append("]");
        return sb.toString();
    }

    private String toString(double[][][] tensor) {
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

    private String toString(int[] array) {
        if (array == null) return "[]";
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i = 0; i < array.length; i++) {
            if (i > 0) sb.append(",");
            sb.append(array[i]);
        }
        sb.append("]");
        return sb.toString();
    }

    /*
     * GETTERS & SETTERS
     */

    public Network getNetwork() { return network; }
    public void setNetwork(Network network) { this.network = network; }
}
