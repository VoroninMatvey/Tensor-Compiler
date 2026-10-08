#pragma once

namespace mlir {
class MLIRContext;
template <typename T> class OwningOpRef;
class ModuleOp;
} // namespace mlir

namespace Labs {
class ONNX_Graph;

mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context, const ONNX_Graph &graph);
} // namespace Labs