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