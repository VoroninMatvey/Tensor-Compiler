cmake_minimum_required(VERSION 3.28)
project("tensor_compiler")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Protobuf REQUIRED)
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS third_party/onnx.proto)

find_package(MLIR REQUIRED CONFIG)
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})

set(SRC src/driver/main.cpp)
set(PROJECT_ROOT_DEF PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")

add_library(tc_frontend STATIC
    src/frontend/importer.cpp
    src/frontend/shape_inference.cpp
    src/frontend/graph_builder.cpp
    ${PROTO_SRCS}
)

target_include_directories(tc_frontend PUBLIC "include" ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(tc_frontend PRIVATE protobuf::libprotobuf)

add_library(tc_backend STATIC
    src/backend/mlir_gen.cpp
    src/backend/lowering.cpp
)

target_include_directories(tc_backend SYSTEM PUBLIC
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
)
target_link_libraries(tc_backend PUBLIC
    tc_frontend
    MLIRIR
    MLIRFuncDialect
    MLIRArithDialect
    MLIRTensorDialect
    MLIRLinalgDialect
)
target_compile_definitions(tc_backend PUBLIC ${LLVM_DEFINITIONS_LIST})

add_executable(${PROJECT_NAME} ${SRC})
target_link_libraries(${PROJECT_NAME} PRIVATE tc_backend)
target_compile_definitions(${PROJECT_NAME} PRIVATE ${PROJECT_ROOT_DEF})

enable_testing()
add_subdirectory(tests)