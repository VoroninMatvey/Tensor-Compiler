#include "backend/lowering.hpp"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/TensorInferTypeOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include <stdexcept>

namespace {
void register_bufferization_models(mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::tensor::registerInferTypeOpInterfaceExternalModels(registry);
    mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);

    context.appendDialectRegistry(registry);
}
} // namespace

namespace Labs {
void lower_to_llvm(mlir::ModuleOp module) {
    mlir::PassManager pm(module.getContext());
    register_bufferization_models(*module.getContext());

    mlir::bufferization::OneShotBufferizePassOptions buf_opts;
    buf_opts.bufferizeFunctionBoundaries = true;
    buf_opts.functionBoundaryTypeConversion =
        mlir::bufferization::LayoutMapOption::IdentityLayoutMap;

    pm.addPass(mlir::bufferization::createOneShotBufferizePass(buf_opts));
    pm.addPass(mlir::createConvertLinalgToLoopsPass());
    pm.addPass(mlir::memref::createExpandStridedMetadataPass());
    pm.addPass(mlir::createLowerAffinePass());
    pm.addPass(mlir::createSCFToControlFlowPass());
    pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
    pm.addPass(mlir::createArithToLLVMConversionPass());
    pm.addPass(mlir::createConvertControlFlowToLLVMPass());
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());

    if (mlir::failed(pm.run(module)))
        throw std::runtime_error("lowering to llvm failed!");
}
} // namespace Labs