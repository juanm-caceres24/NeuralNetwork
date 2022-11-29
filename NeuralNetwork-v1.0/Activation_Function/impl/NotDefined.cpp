#include <iostream>
#include <activation_function/ActivationFunction.cpp>

using namespace std;

class NotDefined : ActivationFunction {
    public:

        // methods

        /*
         * virtual method implementation
         */
        float applyActivationFunction(float value) {
            return value;
        }
};
