#pragma once

#include <cstddef>
#include <vector>
#include <memory>

/**
 * Owns the tensor's underlying data.
 * Tensor objects access Storage through _storage.
 */
class Storage {
private:
    // Actual container for `Storage` data.
    std::vector<double> _data;

public:
    // Storage constructor
    // Must pass exact type: std::vector<double>
    explicit Storage(std::vector<double> data) : _data(std::move(data)) {}

    // Provides mutable access to _data if needed.
    std::vector<double>& data() { 
        return _data;
    }

    // Provides immutable access to _data if needed.
    const std::vector<double>& data() const{   
        return _data;
    }
};

class Tensor {
private:
    /**
     * Shared ownership of the underlying Storage.
     *
     * Multiple Tensor objects can reference the same Storage, allowing
     * operations such as views/transposes without copying the underlying data.
     */
    std::shared_ptr<Storage> _storage; 
    
    // Number of elements along each tensor dimension.
    std::vector<std::size_t> _shape;

    // Number of storage elements to skip when advancing by one position
    // along each tensor dimension.
    std::vector<std::size_t> _stride;

    // Starting position of this Tensor within the underlying Storage.
    // Useful for taking a view() of a Tensor containing a subset of data.
    // Default value: 0
    std::size_t _storage_offset{0};

    /**
     * View constructor (private).
     *
     * - Shares an existing `Storage` and takes shape/stride/offset exactly
     *   as given, so it can describe non-contiguous layouts.
     * - Only Tensor's own methods (slice, select, ...) create views.
     */
    Tensor(
        std::shared_ptr<Storage> storage, 
        std::vector<std::size_t> shape,
        std::vector<std::size_t> stride,
        std::size_t offset
    )
        : _storage(std::move(storage)), 
          _shape(std::move(shape)), 
          _stride(std::move(stride)),
          _storage_offset(offset) {}

    /**
     * Fills _stride with row-major strides derived from _shape.
     * Only valid for newly created, contiguous `Tensor` objects.
     */
    void setStride();

public:

    /**Tensor Constructor (Default)
     * 
     * - Creates a new `Storage` and `shared_ptr` group.
     * - Used when creating an original (non-view) `Tensor`.
     * - Throws `std::invalid_argument` if shape does not match data size.
     */
    Tensor(
        const std::vector<double> data, 
        const std::vector<std::size_t> shape
    );

    // ---- Simple Accessors ----
    const std::vector<std::size_t>& shape() const{ return _shape; }
    const std::vector<std::size_t>& stride() const { return _stride; }
    std::size_t storageOffset() const { return _storage_offset; }
    std::size_t ndim() const { return _shape.size(); }
    

    // Raw access to whole backing vector
    // For a view this includes elements outside the view and ignores offset/strides.
    // Use `at()` for logical element access.
    const std::vector<double>& data() const { return _storage->data(); };
    
    // ---- Larger Functions (defined in Tensor.cpp) ----
    
    // Total number of elements (product of shape; 1 for a scalar)
    std::size_t numel() const;

    // True if elements are row-major with no gaps in storage.
    bool isContiguous() const;

    /**
     * Converts an nD index to a 1D index position in storage.
     * Throws if the index has the wrong length or is out of range. 
     */
    std::size_t flatIndex(const std::vector<std::size_t>& idx) const;
    
    /** Stride-aware element access
     * 
     * e.g. t.at({1, 2}).
     */
    double& at(const std::vector<std::size_t>& idx);
    const double& at(const std::vector<std::size_t>& idx) const;

    /**
     * View of `dim` restricted to [start, stop) with the given step.
     * 
     * Python semantics: negative start/stop count from the end, and
     *      out-of-range values are clamped. 
     * 
     * Step must be > 0.
     * 
     * Shares storage with this tensor (writes are visible in both).
     * 
     */
    Tensor slice(std::size_t dim, long start, long stop, long step = 1) const;
 
    /**
     * View with `dim` removed, fixed at position `index`
     * (negative index counts from the end). Shares storage.
     */
    Tensor select(std::size_t dim, long index) const;

    /**
     * Display tensor in array bracket form
     */
    void display() const;
    // Helper for `display()`
    void print_rec(std::size_t dim, std::size_t pos) const;

    /**
     * Transpose two dimensions of the tensor.
     * 
     * dim0 and dim1 denote matrix dimension indices (typically last 2 of shape)
     * - dim0 = _shape[size - 2] 
     * - dim1 = _shape[size - 1]
     * 
     * Notice: Provide current shape indices, not the expected shape after transpose nor the size of the dimensions.
     * 
     * **WARNING** : returned tensor is NOT contiguous
     */
    Tensor transpose(int dim0, int dim1) const;

    /**
     * Convert a 1D tensor (vector) to a 2D tensor (matrix)
     * 
     * - `M` and `N` are the dimension sizes of the resulting matrix.
     * - The product `M * N` must equal the number of elements in the original tensor.
     *
     * Note: A tensor with shape (M, 1) or (1, N) is a vector in matrix form
     */
    Tensor vectorToMatrix() const;
};
