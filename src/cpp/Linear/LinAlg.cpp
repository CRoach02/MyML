#include "LinAlg.hpp"

#include <cstddef>
#include <stdexcept>

namespace linalg {
    double dot(const Tensor& tensor1, const Tensor& tensor2) {
        if (tensor1.ndim() != 1 || tensor2.ndim() != 1) {
            throw std::invalid_argument("[dot] Both tensors must be 1D (a vector).");
        }
        if (tensor1.shape()[0] != tensor2.shape()[0]) {
            throw std::invalid_argument("[dot] The provided tensors have unequal length.");
        }

        double sum = 0.0;
        for (std::size_t i = 0; i < tensor1.shape()[0]; ++i) {
            sum += tensor1.at({i}) * tensor2.at({i});
        }
        return sum;
    }

    Tensor matmul(const Tensor& tensor1, const Tensor& tensor2) {
        // Supported Tensor shapes: (m, n) @ (n, p) -> (m, p). 
        //      A col vector is a (n, 1) matrix, and a row vector is a (1, n) matrix.
        if (tensor1.ndim() != 2 || tensor2.ndim() != 2) {
            throw std::invalid_argument("[matmul] Both tensors must be 2D (matrices).");
        }
        const std::size_t M = tensor1.shape()[0];
        const std::size_t N1 = tensor1.shape()[1];
        const std::size_t N2 = tensor2.shape()[0];
        const std::size_t P = tensor2.shape()[1];

        if (N1 != N2) {
            throw std::invalid_argument("[matmul] Incompatible inner dimensions for matrix multiplication.");
        }

        Tensor result(std::vector<double>(M * P, 0.0), {M, P});

        for (std::size_t i = 0; i < M; ++i) {
            for (std::size_t j = 0; j < P; ++j) {
                double sum = 0.0;
                for (std::size_t k = 0; k < N1; ++k) {
                    sum += tensor1.at({i, k}) * tensor2.at({k, j});
                }
                result.at({i, j}) = sum;
            }
        }
        return result;
    }

    Tensor addBias(const Tensor& tensor, const Tensor& bias) {
        if (tensor.ndim() < 2 || bias.ndim() != 1 || tensor.shape()[1] != bias.shape()[0]) {
            throw std::invalid_argument("[addBias] Incompatible dimensions for adding bias. Tensor must be at least 2D and bias must be 1D with matching size.");
        }
        const std::size_t M = tensor.shape()[0];
        const std::size_t N = tensor.shape()[1];
        Tensor result(std::vector<double>(M * N, 0.0), {M, N});
        for (std::size_t i = 0; i < M; ++i) {
            for (std::size_t j = 0; j < N; ++j) {
                result.at({i, j}) = tensor.at({i, j}) + bias.at({j});
            }
        }
        return result;
    }
}



// Tensor ops: dot, cross, matmul