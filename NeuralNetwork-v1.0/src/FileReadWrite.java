package src;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;

public class FileReadWrite {

    private ArrayList<String> readSetupFile() {
        ArrayList<String> strFileSetup = new ArrayList<>();
        // open the file
        try {
            FileInputStream fstream = new FileInputStream("config.txt");
            BufferedReader br = new BufferedReader(new InputStreamReader(fstream));
            String strLine;
            // read file line by line and load the arraylist
            while ((strLine = br.readLine()) != null) {
                strFileSetup.add(strLine);
            }
            fstream.close();
        } catch (IOException e) {
            System.out.printf("Failed to open config.txt file\n");
            e.printStackTrace();
        }
        return strFileSetup;
    }

    // --COMPLETE IMPLEMENTATION--

    public ArrayList<LayerSetupFileStruct> readNeuralNetworkFromFile() {
        ArrayList<LayerSetupFileStruct> fileSetup = new ArrayList<>();
        ArrayList<String> strFileSetup = readSetupFile();
        LayerSetupFileStruct fileLayerSetup;
        int operation = -1;
        for (String string : strFileSetup) {
            ArrayList<Double> biases = new ArrayList<>();
            ArrayList<ArrayList<Double>> previousWeights = new ArrayList<>();
            String strActivationFunction = null;
            switch (operation) {
                case 1:
                    strActivationFunction = string ;
                    break;
                case 2:
                    
                    break;
            }
            fileLayerSetup = new LayerSetupFileStruct(biases, previousWeights, strActivationFunction);
            switch (string) {
                case "activationFunction=":
                    operation = 1;
                    break;
                case "previousWeights=":
                    operation = 2;
                    break;
                default:
                    break;
            }
        }
        return fileSetup;
    }

    public void saveNeuralNetworkIntoFile() {

    }
}
