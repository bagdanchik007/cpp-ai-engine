cmake_minimum_required(VERSION 3.20)

project(
    cppai
    VERSION 0.1.0
    DESCRIPTION "A C++ AI and machine learning library"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(cppai)

target_sources(cppai
    PRIVATE
        src/tensor/tensor_shape.cpp
        src/tensor/tensor_storage.cpp
        src/tensor/tensor.cpp
        src/tensor/tensor_operations.cpp
        src/autograd/operation.cpp
        src/autograd/computation_graph.cpp
        src/autograd/variable.cpp
        src/autograd/backward.cpp
        src/nn/parameter.cpp
        src/nn/module.cpp
        src/nn/sequential.cpp
        src/nn/layers/linear.cpp
        src/nn/layers/embedding.cpp
        src/nn/activations/relu.cpp
        src/nn/activations/sigmoid.cpp
        src/nn/activations/softmax.cpp
        src/optim/optimizer.cpp
        src/optim/sgd.cpp
        src/optim/adam.cpp
)

target_include_directories(cppai
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(cppai_console
    apps/console/main.cpp
)

target_link_libraries(cppai_console
    PRIVATE
        cppai
)

enable_testing()

add_subdirectory(tests)