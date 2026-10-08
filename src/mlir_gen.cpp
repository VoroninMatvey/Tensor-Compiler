#include "mlir_gen.hpp"
#include "graph.hpp"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

namespace Labs {
mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context, const ONNX_Graph &graph) {
    mlir::Location loc = mlir::UnknownLoc::get(&context);
    mlir::ModuleOp module = mlir::ModuleOp::create(loc);
    return module;
}
} // namespace Labs