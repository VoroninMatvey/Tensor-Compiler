cmake_minimum_required(VERSION 3.28)
project("Tensor_Compiler")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Protobuf REQUIRED)
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS third_party/onnx.proto)

set(PROJECT_ROOT_DEF PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")

set(SRCS src/main.cpp src/importer.cpp src/shape_inference.cpp src/graph_builder.cpp)
add_executable(${PROJECT_NAME} ${SRCS} ${PROTO_SRCS})

target_include_directories(${PROJECT_NAME} PUBLIC "include" ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(${PROJECT_NAME} PUBLIC protobuf::libprotobuf)
target_compile_definitions(${PROJECT_NAME} PRIVATE ${PROJECT_ROOT_DEF})
