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

include(GoogleTest)

gtest_discover_tests(tensor_tests)