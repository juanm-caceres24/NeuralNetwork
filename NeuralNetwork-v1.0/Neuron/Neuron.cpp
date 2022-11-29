#include <Neuron.h>

using namespace std;

// constructors

/*
 * constructor without parameters
 */
Neuron :: Neuron(void) {
    this->neuronID = NULL;
    this->value = NULL;
    this->bias = NULL;
    this->neuronType = NEURON_TYPE_NOT_DEFINED;
    this->activationFunction = { };
    this->previousNeurons = { };
    this->nextNeurons = { };
}

/*
 * constructor with parameters
 */
Neuron :: Neuron(float neuronID, float bias, ActivationFunction* activationFunction, NeuronTypeEnum neuronType) {
    this->neuronID = neuronID;
    this->value = NULL;
    this->bias = bias;
    this->neuronType = neuronType;
    this->activationFunction = activationFunction;
    this->previousNeurons = { };
    this->nextNeurons = { };
}

// methods

void Neuron :: calculateValue(void) {
    for (NeuralConnection it : previousNeurons)
        value += *it.neuron->getValue() * it.weight;
    value = activationFunction->applyActivationFunction(value);
}

// getters & setters

int* Neuron :: getNeuronID(void) {
    return &neuronID;
}

void Neuron :: setNeuronID(float neuronID) {
    this->neuronID = neuronID;
}

float* Neuron :: getValue(void) {
    return &value;
}

void Neuron :: setValue(float value) {
    this->value = value;
}

float* Neuron :: getBias(void) {
    return &bias;
}

void Neuron :: setBias(float bias) {
    this->bias = bias;
}

NeuronTypeEnum* Neuron :: getNeuronType(void) {
    return &neuronType;
}

void Neuron :: setNeuronType(NeuronTypeEnum neuronType) {
    this->neuronType = neuronType;
}

ActivationFunction* Neuron :: getActivationFunction(void) {
    return activationFunction;
}

void Neuron :: setActivationFunction(ActivationFunction* activationFunction) {
    this->activationFunction = activationFunction;
}

list<NeuralConnection>* Neuron :: getPreviousNeurons(void) {
    return &previousNeurons;
}

void Neuron :: setPreviousNeurons(list<NeuralConnection> previousNeurons) {
    this->previousNeurons = previousNeurons;
}

list<NeuralConnection>* Neuron :: getNextNeurons(void) {
    return &nextNeurons;
}

void Neuron :: setNextNeurons(list<NeuralConnection> nextNeurons) {
    this->nextNeurons = nextNeurons;
}
