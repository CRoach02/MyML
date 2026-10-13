#pragma once

#include "Tensor.hpp"

namespace linalg {
    /**
     * @brief Computes the dot product of two vectors (1D tensors).
     *
     * Tensors must be one dimensional and equal length.
     *  
     * - `tensor.ndim() == 1`
     * 
     * - `tensor1.data().size() == tensor2.data().size()`
     *
     * @param tensor1 First vector.
     * @param tensor2 Second vector.
     * @return The dot product of the two vectors (a scalar).
     * 
     */
    double dot(const Tensor& tensor1, const Tensor& tensor2);
    
    /**
     * @brief Computes the matrix product of two matrices (2D tensors).
     * 
     * Matrices must share equal inner-dimension size:
     * 
     * - Tensor1 shape: (M, N)
     * 
     * - Tensor2 shape: (N, P)
     * 
     * @param tensor1 First matrix (M, N).
     * @param tensor2 Second matrix (N, P).
     * @return A new (M, P) tensor.
     */
    Tensor matmul(const Tensor& tensor1, const Tensor& tensor2);

    /**
     * @brief Adds a bias vector to every row of a matrix.
     *
     * - Tensor shape: (M, N)
     *
     * - Bias shape: (N)
     *
     * @param tensor Matrix of shape (M, N).
     * @param bias Vector of length N.
     * @return A new (M, N) tensor.
     */
    Tensor addBias(const Tensor& tensor, const Tensor& bias);
}
