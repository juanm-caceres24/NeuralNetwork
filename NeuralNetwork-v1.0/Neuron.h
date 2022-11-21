#ifndef NEURONDB_NEURON_H
#define NEURONDB_NEURON_H

#include <iostream>
#include "list"
#include "ActivationFunctionEnum.cpp"
#include "NeuronTypeEnum.cpp"

using namespace std;

class Neuron {
    private:
        float value;
        float bias;
        ActivationFunctionEnum activationFunction;
        NeuronTypeEnum neuronType;
        list<Neuron *> connectedPrevoiusNeurons;
        list<float> weights;
    public:
        Neuron(void);
};

#endif
