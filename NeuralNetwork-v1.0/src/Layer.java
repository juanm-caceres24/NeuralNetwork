package src;

import java.util.ArrayList;

public class Layer{
    
    private ArrayList<Neuron> neurons;
    private ArrayList<Thread> neuronsThreads;

    public Layer() {
        neurons = new ArrayList<>();
        neuronsThreads = new ArrayList<>();
    }

    public ArrayList<Neuron> getNeurons() {
        return neurons;
    }

    public ArrayList<Thread> getNeuronsThreads() {
        return neuronsThreads;
    }
}
