cmake_minimum_required(VERSION 3.28)
project("tensor_compiler")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Protobuf REQUIRED)
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS third_party/onnx.proto)

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

add_executable(${PROJECT_NAME} ${SRC})
target_link_libraries(${PROJECT_NAME} PRIVATE tc_frontend)
target_compile_definitions(${PROJECT_NAME} PRIVATE ${PROJECT_ROOT_DEF})

add_subdirectory(tests)