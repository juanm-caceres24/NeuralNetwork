package src;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

public class FileReadWrite {

    private ArrayList<String> readLinesFromFile( String fileName) {
        ArrayList<String> lines = new ArrayList<>();

        // open the file for reading
        try {
            FileReader reader = new FileReader(fileName);
            BufferedReader br = new BufferedReader(reader);
            String strLine;

            // read each line and load the arraylist
            while ((strLine = br.readLine()) != null) {
                lines.add(strLine);
            }
            br.close();
            reader.close();
        } catch (IOException e) {
            System.out.printf("Failed to open config.txt file\n");
            e.printStackTrace();
        }
        return lines;
    }

    private void writeLinesIntoFile(ArrayList<String> lines, String fileName) {
        
        // open the file for writing
        try {
            FileWriter writer = new FileWriter(fileName);
            BufferedWriter bw = new BufferedWriter(writer);
            
            // write each line to the file
            for (String strLine : lines) {
                bw.write(strLine);
                bw.newLine();
            }
            bw.close();
            writer.close();
        } catch (IOException e) {
            System.out.printf("Failed to open config.txt file\n");
            e.printStackTrace();
        }
    }

    private ArrayList<String> splitLines() {
        ArrayList<String> convertedLines = new ArrayList<>();
        ArrayList<String> lines = readLinesFromFile("config.txt");
        String auxStr;
        for (String string : lines) {
            int i = 0;
            while (i < string.length()) {
                auxStr = "";
                while (i < string.length() && string.charAt(i) != '=' && string.charAt(i) != ',') {
                    auxStr += string.charAt(i);
                    i++;
                }
                convertedLines.add(auxStr);
                i++;
            }
        }
        return convertedLines;
    }

    public ArrayList<LayerFile> readNeuralNetworkFromFile() {
        ArrayList<LayerFile> fileSetup = new ArrayList<>();
        ArrayList<String> convertedLines = splitLines();
        String operation = "";
        for (String string : convertedLines) {
            if (string.equals("layer") || string.equals("activationFunction") || string.equals("neuron") || string.equals("bias") || string.equals("previousWeights")) {
                operation = string;
            } else {
                switch (operation) {
                    case "layer": {
                        fileSetup.add(new LayerFile());
                        break;
                    }
                    case "activationFunction": {
                        fileSetup.get(fileSetup.size() - 1).setActivationFunction(string);
                        break;
                    }
                    case "neuron": {
                        fileSetup.get(fileSetup.size() - 1).biases.add(null);
                        fileSetup.get(fileSetup.size() - 1).previousWeights.add(new ArrayList<Double>());
                        break;
                    }
                    case "bias": {
                        fileSetup.get(fileSetup.size() - 1).biases.set(fileSetup.get(fileSetup.size() - 1).biases.size() - 1, Double.parseDouble(string));
                        break;
                    }
                    case "previousWeights": {
                        if (!string.equals("null")) {
                            fileSetup.get(fileSetup.size() - 1).previousWeights.get(fileSetup.get(fileSetup.size() - 1).previousWeights.size() - 1).add(Double.parseDouble(string));
                        } else {
                            fileSetup.get(fileSetup.size() - 1).previousWeights.get(fileSetup.get(fileSetup.size() - 1).previousWeights.size() - 1).add(null);
                        }
                        break;
                    }
                }
            }
        }
        return fileSetup;
    }

    public void saveNeuralNetworkIntoFile(ArrayList<LayerFile> setupFile) {
        ArrayList<String> lines = new ArrayList<>();

        // loop all the layers of the setupFile
        int i = 0;
        for (LayerFile layerSetupFile : setupFile) {
            lines.add("layer=" + i);
            lines.add("activationFunction=" + layerSetupFile.activationFunction.getClass().getSimpleName().toUpperCase());

            // loop all the neurons of the current layer
            for (int j = 0; j < layerSetupFile.biases.size(); j++) {

                // load the neuron position
                lines.add("neuron=" + j);

                // load the bias
                lines.add("bias=" + layerSetupFile.biases.get(j));

                // load the previousWeights
                String auxLine = "previousWeights=";
                if (layerSetupFile.previousWeights.get(j).size() != 0) {
                    for (Double previousWeight : layerSetupFile.previousWeights.get(j)) {
                        auxLine += previousWeight + ",";
                    }
                    auxLine = auxLine.substring(0, auxLine.length() - 1);
                } else {
                    auxLine += "null";
                }
                lines.add(auxLine);
            }
            i++;
        }

        // save lines into file
        writeLinesIntoFile(lines, "config.txt");
    }

    public ArrayList<Double> readInputFromFile() {
        ArrayList<String> lines = readLinesFromFile("input.txt");
        ArrayList<Double> values = new ArrayList<>();
        for (String line : lines) {
            values.add(Double.parseDouble(line));
        }
        return values;
    }
}
