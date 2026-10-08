cmake_minimum_required(VERSION 3.28)
project("tensor_compiler")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Protobuf REQUIRED)
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS third_party/onnx.proto)

find_package(MLIR REQUIRED CONFIG)
separate_arguments(LLVM_DEFENITIONS_LIST NATIVE_COMMAND ${LLVM_DEFENITIONS})

set(SRC src/main.cpp)
set(PROJECT_ROOT_DEF PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")

add_library(tc_frontend STATIC
    src/importer.cpp
    src/shape_inference.cpp
    src/graph_builder.cpp
    ${PROTO_SRCS}
)

target_include_directories(tc_frontend PUBLIC "include" ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(tc_frontend PRIVATE protobuf::libprotobuf)

add_library(tc_mlirgen STATIC
    src/mlir_gen.cpp
)

target_include_directories(tc_mlirgen SYSTEM PUBLIC
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
)
target_link_libraries(tc_mlirgen PUBLIC
    tc_frontend
    MLIRIR
    MLIRFuncDialect
    MLIRArithDialect
    MLIRTensorDialect
    MLIRLinalgDialect
)
target_compile_definitions(tc_mlirgen PUBLIC ${LLVM_DEFENITIONS_LIST})

add_executable(${PROJECT_NAME} ${SRC})
target_link_libraries(${PROJECT_NAME} PRIVATE tc_mlirgen)
target_compile_definitions(${PROJECT_NAME} PRIVATE ${PROJECT_ROOT_DEF})

enable_testing()
add_subdirectory(tests)