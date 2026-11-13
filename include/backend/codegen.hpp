#pragma once
#include <filesystem>
#include <memory>

namespace llvm {
class Module;
class LLVMContext;
class TargetMachine;
enum class CodeGenFileType;
} // namespace llvm

namespace mlir {
class ModuleOp;
} // namespace mlir

namespace Labs {
std::unique_ptr<llvm::Module> translate_to_llvm_ir(mlir::ModuleOp module,
                                                   llvm::LLVMContext &llvm_context);
void optimize_llvm_ir(llvm::Module &module, llvm::TargetMachine &tm, int opt_level = 2);
void emit_target_code(llvm::Module &module, llvm::TargetMachine &tm,
                      const std::filesystem::path path, llvm::CodeGenFileType file_type);
std::unique_ptr<llvm::TargetMachine> create_target_machine();
void init_llvm_targets();
} // namespace Labs