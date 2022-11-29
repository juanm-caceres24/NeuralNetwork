#ifndef NEURONDB_NEURON_H

#define NEURONDB_NEURON_H

#include <iostream>
#include <list>
#include <NeuralConnection.cpp>
#include <Activation_Function/ActivationFunction.cpp>
#include <Activation_Function/ActivationFunctionEnum.cpp>
#include <NeuronTypeEnum.cpp>

using namespace std;

class Neuron {
    private:

        // attributes
        int neuronID;
        float value;
        float bias;
        ActivationFunction* activationFunction;
        NeuronTypeEnum neuronType;
        list<NeuralConnection> previousNeurons;
        list<NeuralConnection> nextNeurons;

        // methods

    public:

        // constructors
        Neuron(void);
        Neuron(float neuronID, float bias, ActivationFunction* activationFunction, NeuronTypeEnum neuronType);

        // methods
        void calculateValue(void);

        // getters & setters
        int* getNeuronID(void);
        void setNeuronID(float neuronID);
        float* getValue(void);
        void setValue(float value);
        float* getBias(void);
        void setBias(float bias);
        NeuronTypeEnum* getNeuronType(void);
        void setNeuronType(NeuronTypeEnum neuronType);
        ActivationFunction* getActivationFunction(void);
        void setActivationFunction(ActivationFunction* activationFunction);
        list<NeuralConnection>* getPreviousNeurons(void);
        void setPreviousNeurons(list<NeuralConnection> previousNeurons);
        list<NeuralConnection>* getNextNeurons(void);
        void setNextNeurons(list<NeuralConnection> nextNeurons);
};

#endif
