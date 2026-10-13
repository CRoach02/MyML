#pragma once

#include "Activations.hpp"
#include "../Linear/Tensor.hpp"

#include <vector>

namespace model {
    // For now this is a namespace for model functions
    //      may be converted to a class later if needed
    Tensor layer(
            const Tensor& input, 
            const Tensor& weights, 
            const Tensor& bias, 
            Activations::Type activation_function
        );
}