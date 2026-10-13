#include "../cpp/Linear/LinAlg.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

void bind_linalg(nb::module_& m) {
    m.def("dot", &linalg::dot);
}
