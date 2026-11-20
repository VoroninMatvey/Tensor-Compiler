#include "backend/codegen.hpp"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"
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

#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>

namespace Labs {

namespace {
void emit_packed_wrapper(llvm::Module &module) {
    llvm::Function *ciface = module.getFunction("_mlir_ciface_forward");
    if (!ciface)
        throw std::runtime_error("Function _mlir_ciface_forward not found");

    llvm::LLVMContext &ctx = module.getContext();
    llvm::Type *ptr_type = llvm::PointerType::getUnqual(ctx);
    llvm::Type *void_type = llvm::Type::getVoidTy(ctx);

    llvm::SmallVector<llvm::Type *> param_types = {ptr_type};
    llvm::FunctionType *fn_type = llvm::FunctionType::get(void_type, param_types, false);
    llvm::Function *packed_fn =
        llvm::Function::Create(fn_type, llvm::Function::ExternalLinkage, "forward_packed", module);

    llvm::BasicBlock *entry_block = llvm::BasicBlock::Create(ctx, "entry", packed_fn);
    llvm::IRBuilder<> builder(entry_block);
    llvm::Value *args_base = packed_fn->getArg(0);

    std::size_t num_ciface_args = ciface->arg_size();
    llvm::SmallVector<llvm::Value *, 5> call_args;
    call_args.reserve(num_ciface_args);

    for (std::size_t i = 0; i < num_ciface_args; ++i) {
        llvm::Value *index_value = builder.getInt64(i);
        llvm::Value *slot_ptr =
            builder.CreateInBoundsGEP(ptr_type, args_base, index_value, "slot_ptr");

        llvm::Value *arg_value = builder.CreateLoad(ptr_type, slot_ptr, "arg_value");
        call_args.push_back(arg_value);
    }

    builder.CreateCall(ciface, call_args);
    builder.CreateRetVoid();
}
} // namespace

std::unique_ptr<llvm::Module> translate_to_llvm_ir(mlir::ModuleOp module,
                                                   llvm::LLVMContext &llvm_context) {
    mlir::registerBuiltinDialectTranslation(*module.getContext());
    mlir::registerLLVMDialectTranslation(*module.getContext());

    std::unique_ptr<llvm::Module> llvm_module = mlir::translateModuleToLLVMIR(module, llvm_context);

    if (!llvm_module)
        llvm::errs() << "Failed to emit LLVM IR\n";

    // Emit universal packed ABI wrapper for mlir_ciface_forward
    emit_packed_wrapper(*llvm_module);
    return llvm_module;
}

void init_llvm_targets() {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmParser();
    llvm::InitializeNativeTargetAsmPrinter();
}

std::unique_ptr<llvm::TargetMachine> create_target_machine(int opt_level) {
    llvm::Triple triple(llvm::sys::getDefaultTargetTriple());

    std::string error;
    const llvm::Target *target = llvm::TargetRegistry::lookupTarget(triple, error);
    if (!target)
        throw std::runtime_error("cannot find target for: " + triple.str() + ": " + error);

    std::string cpu = llvm::sys::getHostCPUName().str();
    llvm::SubtargetFeatures features;
    for (const auto &[name, enabled] : llvm::sys::getHostCPUFeatures())
        features.AddFeature(name, enabled);

    llvm::CodeGenOptLevel gen_opt_level;
    switch (opt_level) {
    case 0:
        gen_opt_level = llvm::CodeGenOptLevel::None;
        break;
    case 1:
        gen_opt_level = llvm::CodeGenOptLevel::Less;
        break;
    case 2:
        gen_opt_level = llvm::CodeGenOptLevel::Default;
        break;
    case 3:
        gen_opt_level = llvm::CodeGenOptLevel::Aggressive;
        break;
    default:
        throw std::runtime_error("Invalid optimization level: " + std::to_string(opt_level));
    }

    llvm::TargetOptions options;
    options.AllowFPOpFusion = llvm::FPOpFusion::Fast;
    return std::unique_ptr<llvm::TargetMachine>(
        target->createTargetMachine(triple, cpu, features.getString(), options, llvm::Reloc::PIC_,
                                    std::nullopt, gen_opt_level));
}

void optimize_llvm_ir(llvm::Module &module, llvm::TargetMachine &tm, int opt_level) {
    module.setTargetTriple(tm.getTargetTriple());
    module.setDataLayout(tm.createDataLayout());

    if (opt_level == 0)
        return;

    llvm::OptimizationLevel level;
    switch (opt_level) {
    case 1:
        level = llvm::OptimizationLevel::O1;
        break;
    case 2:
        level = llvm::OptimizationLevel::O2;
        break;
    case 3:
        level = llvm::OptimizationLevel::O3;
        break;
    default:
        throw std::runtime_error("Invalid optimization level: " + std::to_string(opt_level));
    }

    llvm::LoopAnalysisManager lam;
    llvm::FunctionAnalysisManager fam;
    llvm::CGSCCAnalysisManager cgam;
    llvm::ModuleAnalysisManager mam;

    llvm::PipelineTuningOptions pto;
    pto.LoopVectorization = opt_level > 1;
    pto.SLPVectorization = opt_level > 1;

    llvm::PassBuilder pb(&tm, pto);
    pb.registerLoopAnalyses(lam);
    pb.registerFunctionAnalyses(fam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerModuleAnalyses(mam);
    pb.crossRegisterProxies(lam, fam, cgam, mam);

    llvm::ModulePassManager mpm = pb.buildPerModuleDefaultPipeline(level);
    mpm.run(module, mam);

    if (llvm::verifyModule(module, &llvm::errs()))
        throw std::runtime_error("LLVM Moule was broken after optimization");
}

void emit_target_code(llvm::Module &module, llvm::TargetMachine &tm,
                      const std::filesystem::path &path, llvm::CodeGenFileType file_type) {
    std::error_code ec;
    auto flags = (file_type == llvm::CodeGenFileType::AssemblyFile) ? llvm::sys::fs::OF_Text
                                                                    : llvm::sys::fs::OF_None;

    llvm::raw_fd_ostream dest(path.string(), ec, flags);
    if (ec)
        throw std::runtime_error("Could not open file " + path.string() + ": " + ec.message());

    llvm::legacy::PassManager codegen_pm;
    if (tm.addPassesToEmitFile(codegen_pm, dest, nullptr, file_type))
        throw std::runtime_error("TargetMachine can't emit a file of this type");

    codegen_pm.run(module);
    dest.flush();
}

void emit_shared_library(llvm::Module &module, llvm::TargetMachine &tm,
                         const std::filesystem::path &so_path) {
    llvm::SmallString<256> temp_obj_buf;
    // Creates a temporary file from a prefix, a random suffix, and a dot with the second argument
    std::error_code ec = llvm::sys::fs::createTemporaryFile("tc_model", "o", temp_obj_buf);

    if (ec)
        throw std::runtime_error("Cannot create temporaary object file:" + ec.message());
    std::filesystem::path temp_obj_path(temp_obj_buf.str().str());

    struct TempFileGuard {
        std::filesystem::path path_;
        ~TempFileGuard() {
            std::error_code remove_ec;
            // Overloading with 2 arguments don't throw exceptions!
            std::filesystem::remove(path_, remove_ec);
        }
    } guard{temp_obj_path};

    emit_target_code(module, tm, temp_obj_path, llvm::CodeGenFileType::ObjectFile);

    llvm::ErrorOr<std::string> linker = llvm::sys::findProgramByName("cc");
    if (!linker)
        linker = llvm::sys::findProgramByName("gcc");
    if (!linker)
        linker = llvm::sys::findProgramByName("clang");
    if (!linker)
        throw std::runtime_error("Cannot find a system linker driver (cc, clang, or gcc)");

    // Args in ExecuteAndWait require llvm::ArrauRef<llvm::StringRef> so create std::string values
    std::string temp_obj_str = temp_obj_path.string();
    std::string so_path_str = so_path.string();
    std::string runtime_dir = MLIR_RUNTIME_LIB_DIR;
    std::string lib_dir_flag = "-L" + runtime_dir;
    std::string rpath_flag = "-Wl,-rpath," + runtime_dir;

    llvm::SmallVector<llvm::StringRef, 10> args = {
        *linker, "-shared", temp_obj_str, lib_dir_flag, rpath_flag, "-lmlir_c_runner_utils",
        "-lm",   "-o",      so_path_str};

    std::string err_msg;
    int rc = llvm::sys::ExecuteAndWait(*linker, args, std::nullopt, {}, 0, 0, &err_msg);

    if (rc != 0)
        throw std::runtime_error("Failed to link shared library '" + so_path_str + "' (exit code " +
                                 std::to_string(rc) + ")" +
                                 (err_msg.empty() ? "" : ": " + err_msg));
}

} // namespace Labs