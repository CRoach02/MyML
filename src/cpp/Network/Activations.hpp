#pragma once

#include "../Linear/Tensor.hpp"


namespace Activations {
    /**
     * Names for respective activation functions.
     * Supported functions:
     * - ReLU
     * - Sigmoid
     * - TanH
     * - None (default case == Linear)
     */
    enum class Type {None, ReLU, Sigmoid, TanH};
    Tensor relu(const Tensor& tensor);
    Tensor sigmoid(const Tensor& tensor);
    Tensor tanh(const Tensor& tensor);

    Tensor apply(const Tensor& tensor, Type type);
}