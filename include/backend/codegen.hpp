#pragma once
#include <memory>

namespace llvm {
class Module;
class LLVMContext;
} // namespace llvm

namespace mlir {
class ModuleOp;
} // namespace mlir

namespace Labs {
std::unique_ptr<llvm::Module> translate_to_llvm_ir(mlir::ModuleOp module,
                                                   llvm::LLVMContext &llvm_context);
} // namespace Labs