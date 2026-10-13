#include "Tensor.hpp"

#include <algorithm>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>

// ---- Constructor(s) ----

Tensor::Tensor(std::vector<double> data, std::vector<std::size_t> shape) 
    :   _storage(std::make_shared<Storage>(std::move(data))),
        _shape(std::move(shape)),
        _stride(_shape.size(), 0)   // _shape is initialized before _stride
    {
        if (numel() != _storage->data().size()) {
            throw std::invalid_argument("[Tensor] Shape does not match the number of data elements.");
        }
        setStride();
    }


void Tensor::setStride() {
    const std::size_t n = _shape.size();
    if (n == 0) {
        return;
    }
    _stride[n - 1] = 1;

    for (std::size_t i = n - 1; i > 0; --i) {
        _stride[i - 1] = _shape[i] * _stride[i];
    }
}

// ---- Shape / Layout queries ----

std::size_t Tensor::numel() const {
    return std::accumulate(
        _shape.begin(), 
        _shape.end(),
        std::size_t{1},
        std::multiplies<>()
    );
}


bool Tensor::isContiguous() const {
    std::size_t expected = 1;
    for (std::size_t i = _shape.size(); i > 0; --i) {
        if (_shape[i-1] != 1 && _stride[i-1] != expected) {
            return false;
        }
        expected *= _shape[i-1];
    }
    return true;
}

// ---- Element access ----

std::size_t Tensor::flatIndex(const std::vector<std::size_t>& idx) const {
    if (idx.size() != _shape.size()) {
        throw std::invalid_argument("[flatIndex] Provided index amount exceeds storage size.");
    }
    std::size_t pos = _storage_offset;
    for (std::size_t i = 0; i < idx.size(); ++i) {
        if (idx[i] >= _shape[i]) {
            throw std::out_of_range("[flatIndex] Index is out of range.");
        }
        pos += idx[i] * _stride[i];
    }
    return pos;
}


double& Tensor::at(const std::vector<std::size_t>& idx) {
    return _storage->data()[flatIndex(idx)];
}


const double& Tensor::at(const std::vector<std::size_t>& idx) const {
    return _storage->data()[flatIndex(idx)];
}

// ---- Views ----

Tensor Tensor::slice(std::size_t dim, long start, long stop, long step) const {
    if (dim >= _shape.size()) {
        throw std::out_of_range("[slice] Dimension is out of range.");
    }
    if (step <= 0) {
        throw std::invalid_argument("[slice] Step must be > 0.");
    }

    const long n = static_cast<long>(_shape[dim]);

    // Pythonic negative indices
    if (start < 0) start += n;
    if (stop < 0) stop += n;

    // Clamp
    start = std::max(0L, std::min(start, n));
    stop = std::max(start, std::min(stop, n));

    std::vector<std::size_t> new_shape = _shape;
    std::vector<std::size_t> new_stride = _stride;

    new_shape[dim] = static_cast<std::size_t>((stop - start + step - 1) / step);
    new_stride[dim] = _stride[dim] * static_cast<std::size_t>(step);
    std::size_t new_offset = _storage_offset + static_cast<std::size_t>(start) * _stride[dim];

    // Same _storage pointer: this is what makes it a view
    return Tensor(_storage, std::move(new_shape), std::move(new_stride), new_offset);
}


Tensor Tensor::select(std::size_t dim, long index) const {
    if (dim >= _shape.size()) {
        throw std::out_of_range("[select] Dimension is out of range.");
    }

    const long n = static_cast<long>(_shape[dim]);
    if (index < 0) index += n;
    if (index < 0 || index >= n) {
        throw std::out_of_range("[select] Index is out of range.");
    }
    
    std::vector<std::size_t> new_shape = _shape;
    std::vector<std::size_t> new_stride = _stride;
    new_shape.erase(new_shape.begin() + static_cast<std::ptrdiff_t>(dim));
    new_stride.erase(new_stride.begin() + static_cast<std::ptrdiff_t>(dim));

    std::size_t new_offset = _storage_offset + static_cast<std::size_t>(index) * _stride[dim];

    return Tensor(_storage, std::move(new_shape), std::move(new_stride), new_offset);
}


Tensor Tensor::transpose(int dim0, int dim1) const {
    int n = static_cast<int>(_shape.size());
    // support negative dims, e.g. -2, -1
    //      note: avoid using negative dims if possible
    if (dim0 < 0) dim0 += n;
    if (dim1 < 0) dim1 += n;
    if (dim0 < 0 || dim0 >= n || dim1 < 0 || dim1 >= n)
        throw std::out_of_range("[transpose] Dimension is out of range.");

    // Copy shape/stride/offset, share underlying storage
    Tensor out = *this;
    std::swap(out._shape[dim0], out._shape[dim1]);
    std::swap(out._stride[dim0], out._stride[dim1]);
    return out;
}


Tensor Tensor::vectorToMatrix() const {
    if (_shape.size() != 1) {
        throw std::invalid_argument("[vectorToMatrix] Tensor must be 1D (vector).");
    }
    std::vector<std::size_t> new_shape = {_shape[0], 1};
    std::vector<std::size_t> new_stride = {_stride[0], 1};
    return Tensor(_storage, std::move(new_shape), std::move(new_stride), _storage_offset);
}

// ---- Tensor display ----

void Tensor::print_rec(std::size_t dim, std::size_t pos) const {
    std::cout << '[';
    for (std::size_t i = 0; i < _shape[dim]; ++i) {
        if (i > 0) std::cout << ", ";
        std::size_t next = pos + i * _stride[dim];
        if (dim + 1 == _shape.size())
            std::cout << data()[next];
        else
            print_rec(dim + 1, next);
    }
    std::cout << ']';
}


void Tensor::display() const {
    if (_shape.empty()) {
        std::cout << data()[_storage_offset] << '\n';
        return;
    }
    print_rec(0, _storage_offset);
    std::cout << '\n';
}

