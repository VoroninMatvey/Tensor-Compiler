#pragma once

namespace mlir {
class ModuleOp;
} // namespace mlir

namespace Labs {
// Lowers a module produced by mlirGen down to the LLVM dialect.
// Throws std::runtime_error if any pass fails.
void lower_to_llvm(mlir::ModuleOp module);
} // namespace Labs