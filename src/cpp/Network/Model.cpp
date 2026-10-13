
#include "Model.hpp"
#include "../Linear/LinAlg.hpp"

namespace model {
    
    Tensor layer(
        const Tensor& input, 
        const Tensor& weights, 
        const Tensor& bias, 
        Activations::Type activation_function
    ) {
        Tensor z = linalg::addBias(linalg::matmul(input, weights.transpose(0, 1)), bias);
        return Activations::apply(z, activation_function);
    }

}