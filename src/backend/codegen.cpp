#include "backend/codegen.hpp"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR//LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include <memory>

namespace Labs {
std::unique_ptr<llvm::Module> translate_to_llvm_ir(mlir::ModuleOp module,
                                                   llvm::LLVMContext &llvm_context) {
    mlir::registerBuiltinDialectTranslation(*module.getContext());
    mlir::registerLLVMDialectTranslation(*module->getContext());

    std::unique_ptr<llvm::Module> llvm_module = mlir::translateModuleToLLVMIR(module, llvm_context);

    if (!llvm_module)
        llvm::errs() << "Failed to emit LLVM IR\n";

    return llvm_module;
}
} // namespace Labs