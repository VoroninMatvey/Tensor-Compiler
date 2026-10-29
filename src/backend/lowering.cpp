#include "backend/lowering.hpp"

#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include <stdexcept>

namespace {
void register_bufferization_models(mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
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

    if (mlir::failed(pm.run(module)))
        throw std::runtime_error("lowering to llvm failed!");
}
} // namespace Labs