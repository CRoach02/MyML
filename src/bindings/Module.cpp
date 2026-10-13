#include <nanobind/nanobind.h>

namespace nb = nanobind;

void bind_tensor(nb::module_& m);
void bind_linalg(nb::module_& m);

NB_MODULE(MyMLPy, m) {
    bind_tensor(m);
    bind_linalg(m);
}