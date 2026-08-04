cmake_minimum_required(VERSION 3.28)
project("Tensor_Compiler")
set(CMAKE_CXX_STANDARD 20)

find_package(Protobuf REQUIRED)
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS third_party/onnx.proto)

set(SRCS src/main.cpp)
add_executable(${PROJECT_NAME} ${SRCS} ${PROTO_SRCS})
target_include_directories(${PROJECT_NAME} PUBLIC "include" ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(${PROJECT_NAME} PUBLIC protobuf::libprotobuf)
