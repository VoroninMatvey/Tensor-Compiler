cmake_minimum_required(VERSION 3.28)
project("Tensor_Compiler")
set(CMAKE_CXX_STANDARD 20)

set(SRCS src/main.cpp)
add_executable(${PROJECT_NAME} ${SRCS})
target_include_directories(${PROJECT_NAME} PUBLIC "include")

