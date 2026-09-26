include(FetchContent)

FetchContent_Declare(
    google_tests
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
)
FetchContent_MakeAvailable(google_tests)

add_executable(test_shape_inference test_shape_inference.cpp)
target_link_libraries(test_shape_inference PRIVATE tc_frontend gtest_main)
add_test(NAME ShapeInference COMMAND test_shape_inference)