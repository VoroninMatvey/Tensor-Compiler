include(FetchContent)

FetchContent_Declare(
    google_tests
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
)
FetchContent_MakeAvailable(google_tests)

add_executable(test_shape_inference shape_inference_test.cpp)
target_link_libraries(test_shape_inference PRIVATE tc_frontend gtest_main)
add_test(NAME ShapeInference COMMAND test_shape_inference)

add_executable(test_importer importer_test.cpp)
target_link_libraries(test_importer PRIVATE tc_frontend gtest_main)
target_compile_definitions(test_importer PRIVATE ${PROJECT_ROOT_DEF})
add_test(NAME Importer COMMAND test_importer)