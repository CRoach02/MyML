#include "Activations.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace Activations {
    Tensor relu(const Tensor& tensor) {
        std::size_t M = tensor.shape()[0];
        std::size_t N = tensor.shape()[1];
        Tensor result(std::vector<double>(M * N, 0.0), {M, N});
        for (std::size_t i = 0; i < M; ++i) {
            for (std::size_t j = 0; j < N; ++j) {
                result.at({i, j}) = std::max(0.0, tensor.at({i, j}));
            }
        }
        return result;
    }

    Tensor sigmoid(const Tensor& tensor) {
        std::size_t M = tensor.shape()[0];
        std::size_t N = tensor.shape()[1];
        Tensor result(std::vector<double>(M * N, 0.0), {M, N});
        for (std::size_t i = 0; i < M; ++i) {
            for (std::size_t j = 0; j < N; ++j) {
                result.at({i, j}) = 1.0 / (1.0 + std::exp(-tensor.at({i, j})));
            }
        }
        return result;
    }

    Tensor tanh(const Tensor& tensor) {
        std::size_t M = tensor.shape()[0];
        std::size_t N = tensor.shape()[1];
        Tensor result(std::vector<double>(M * N, 0.0), {M, N});
        for (std::size_t i = 0; i < M; ++i) {
            for (std::size_t j = 0; j < N; ++j) {
                result.at({i, j}) = 
                    (std::exp(tensor.at({i, j})) - std::exp(-tensor.at({i, j}))) / (std::exp(tensor.at({i, j})) + std::exp(-tensor.at({i, j})));
            }
        }
        return result;
    }

    Tensor apply(const Tensor& tensor, Type type) {
        switch (type) {
            case Type::None:    return tensor;
            case Type::ReLU:    return relu(tensor);
            case Type::Sigmoid: return sigmoid(tensor);
            case Type::TanH:    return tanh(tensor);
        }
        throw std::invalid_argument("[apply] Unknown activation type.");
    }

}