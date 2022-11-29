#include <iostream>
#include <Activation_Function/ActivationFunctionEnum.cpp>

using namespace std;

class ActivationFunction {
    protected:

        // attributes
        ActivationFunctionEnum activationFunction;

    public:

        // constructors

        /*
         * constructor whitout parameters
         */
        ActivationFunction(void) {
            activationFunction = ACTIVATION_FUNCTION_NOT_DEFINED;
        };

        /*
         * constructor whit parameters
         */
        ActivationFunction(ActivationFunctionEnum activationFunction) {
            this->activationFunction = activationFunction;
        };

        // methods
        virtual float applyActivationFunction(float value);

        // getters & setters
        ActivationFunctionEnum* getActivationFunction() {
            return &activationFunction;
        };
};
