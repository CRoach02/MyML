#include "../cpp/Linear/Tensor.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;


void bind_tensor(nb::module_& m) {
    nb::class_<Tensor>(m, "Tensor")
        .def(nb::init<std::vector<double>, std::vector<std::size_t>>(),
            nb::arg("data"), nb::arg("shape"))
        .def_prop_ro("shape", &Tensor::shape)
        .def("display", &Tensor::display);
}