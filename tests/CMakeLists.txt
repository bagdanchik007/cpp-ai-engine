include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
)

FetchContent_MakeAvailable(googletest)

add_executable(tensor_tests
    tensor/tensor_tests.cpp
)

target_link_libraries(tensor_tests
    PRIVATE
        cppai
        GTest::gtest_main
)

add_executable(autograd_tests
    autograd/autograd_tests.cpp
)

target_link_libraries(autograd_tests
    PRIVATE
        cppai
        GTest::gtest_main
)

add_executable(nn_tests
    nn/nn_tests.cpp
)

target_link_libraries(nn_tests
    PRIVATE
        cppai
        GTest::gtest_main
)

add_executable(optim_tests
    optim/optim_tests.cpp
)

target_link_libraries(optim_tests
    PRIVATE
        cppai
        GTest::gtest_main
)

add_executable(tokenizer_tests
    tokenizer/tokenizer_tests.cpp
)

target_link_libraries(tokenizer_tests
    PRIVATE
        cppai
        GTest::gtest_main
)

add_executable(core_tests
    core/core_tests.cpp
)

target_link_libraries(core_tests
    PRIVATE
        cppai
        GTest::gtest_main
)

include(GoogleTest)

gtest_discover_tests(tensor_tests)
gtest_discover_tests(autograd_tests)
gtest_discover_tests(nn_tests)
gtest_discover_tests(optim_tests)
gtest_discover_tests(tokenizer_tests)
gtest_discover_tests(core_tests)
