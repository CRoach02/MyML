cmake_minimum_required(VERSION 3.20)

project(MyML LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_package(nanobind CONFIG REQUIRED)

add_executable(MyML
    src/main.cpp
    src/cpp/Linear/Tensor.cpp
    src/cpp/Linear/LinAlg.cpp
    src/cpp/Network/Activations.cpp
    src/cpp/Network/Model.cpp
)

nanobind_add_module(MyMLPy
    # src/bindings/bindings.cpp
    src/bindings/Tensor_Bindings.cpp
    src/cpp/Linear/Tensor.cpp 
    src/cpp/Linear/LinAlg.cpp
    src/bindings/Module.cpp
    src/bindings/LinAlg_Bindings.cpp
)