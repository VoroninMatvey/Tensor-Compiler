cmake_minimum_required(VERSION 3.28)
project("tensor_compiler")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Protobuf REQUIRED)
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS third_party/onnx.proto)

find_package(MLIR REQUIRED CONFIG)
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
llvm_map_components_to_libnames(LLVM_CODEGEN_LIBS native passes)

set(SRC src/driver/main.cpp)
set(PROJECT_ROOT_DEF PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")

if(TARGET mlir_c_runner_utils)
    set(MLIR_RUNTIME_LIB_DIR "$<TARGET_FILE_DIR:mlir_c_runner_utils>")
else()
    set(MLIR_RUNTIME_LIB_DIR "${LLVM_LIBRARY_DIR}")
endif()

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
    src/backend/codegen.cpp
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
    MLIRPass
    MLIRBufferizationTransforms
    MLIRBufferizationPipelines
    MLIRTensorInferTypeOpInterfaceImpl
    MLIRArithTransforms
    MLIRLinalgTransforms
    MLIRTensorTransforms
    MLIRAffineToStandard
    MLIRCAPIConversion
    MLIRMemRefToLLVM
    MLIRArithToLLVM
    MLIRControlFlowToLLVM
    MLIRFuncToLLVM
    MLIRReconcileUnrealizedCasts
    MLIRTargetLLVMIRExport
    MLIRBuiltinToLLVMIRTranslation
    MLIRLLVMToLLVMIRTranslation
    ${LLVM_CODEGEN_LIBS}
)
target_compile_definitions(tc_backend PUBLIC ${LLVM_DEFINITIONS_LIST})
target_compile_definitions(tc_backend PRIVATE MLIR_RUNTIME_LIB_DIR="${MLIR_RUNTIME_LIB_DIR}")
target_compile_options(tc_backend PRIVATE -fno-rtti)

add_executable(${PROJECT_NAME} ${SRC})
target_link_libraries(${PROJECT_NAME} PRIVATE tc_backend)
target_compile_definitions(${PROJECT_NAME} PRIVATE ${PROJECT_ROOT_DEF})
#LLVM compiled without rtti, cl try to use this
target_compile_options(${PROJECT_NAME} PRIVATE -fno-rtti)

add_executable(runner src/driver/runner.cpp)
target_link_libraries(runner PRIVATE tc_frontend ${CMAKE_DL_LIBS})

enable_testing()
add_subdirectory(tests)