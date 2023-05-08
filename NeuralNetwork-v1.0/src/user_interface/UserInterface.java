package src.user_interface;

import java.util.ArrayList;

import src.LayerFile;
import src.Layer;

public interface UserInterface  {
    
    public abstract String createOrLoadNeuralNetwork();
    public abstract ArrayList<LayerFile> createLayerFile(Double INIT_BIAS, Double INIT_WEIGHT);
    public abstract void showLayers(ArrayList<Layer> layers);
    public abstract void showLayer(Layer layer);
    public abstract void clearConsole();
}
