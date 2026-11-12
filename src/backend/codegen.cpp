#include "backend/codegen.hpp"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/SubtargetFeature.h"
#include "llvm/TargetParser/Triple.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR//LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include <memory>
#include <stdexcept>
#include <string>

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

void init_llvm_targets() {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmParser();
    llvm::InitializeNativeTargetAsmPrinter();
}

std::unique_ptr<llvm::TargetMachine> create_target_machine() {
    llvm::Triple triple(llvm::sys::getDefaultTargetTriple());

    std::string error;
    const llvm::Target *target = llvm::TargetRegistry::lookupTarget(triple, error);
    if (!target)
        throw std::runtime_error("cannot find target for: " + triple.str() + ": " + error);

    std::string cpu = llvm::sys::getHostCPUName().str();
    llvm::SubtargetFeatures features;
    for (const auto &[name, enabled] : llvm::sys::getHostCPUFeatures())
        features.AddFeature(name, enabled);

    llvm::TargetOptions options;
    return std::unique_ptr<llvm::TargetMachine>(
        target->createTargetMachine(triple, cpu, features.getString(), options, llvm::Reloc::PIC_));
}
} // namespace Labs