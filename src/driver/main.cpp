#include "backend/codegen.hpp"
#include "backend/lowering.hpp"
#include "backend/mlir_gen.hpp"
#include "frontend/graph.hpp"
#include "frontend/graph_builder.hpp"
#include "frontend/importer.hpp"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>

namespace fs = std::filesystem;

namespace {
enum class EmitAction { MLIR, LLVM, Asm, Obj };

llvm::cl::OptionCategory compiler_category("Tensor Compiler Options");

llvm::cl::opt<std::string> input_filepath(llvm::cl::Positional,
                                          llvm::cl::desc("Path to <input.onnx>"),
                                          llvm::cl::Required, llvm::cl::cat(compiler_category));

llvm::cl::opt<EmitAction> emit_action(
    "emit", llvm::cl::desc("Desired output type"),
    llvm::cl::values(clEnumValN(EmitAction::MLIR, "mlir", "Output lowered MLIR (LLVM dialect)"),
                     clEnumValN(EmitAction::LLVM, "llvm", "Output optimized LLVM IR"),
                     clEnumValN(EmitAction::Asm, "asm", "Output target assembly (.s)"),
                     clEnumValN(EmitAction::Obj, "obj", "Output native object file (.o)")),
    llvm::cl::init(EmitAction::LLVM), llvm::cl::cat(compiler_category));

llvm::cl::opt<int>
    opt_level("O", llvm::cl::Prefix,
              llvm::cl::desc("Optimization levels: -O0, -O1, -O2, -O3. Default: -O2"),
              llvm::cl::init(2), llvm::cl::cat(compiler_category));

llvm::cl::opt<std::string> output_filepath(
    "o",
    llvm::cl::desc("Path to output_file. Mandatory for .o, the rest are optional. Default: stdout"),
    llvm::cl::value_desc("filepath"), llvm::cl::init("-"), llvm::cl::cat(compiler_category));

llvm::cl::opt<bool> draw_dot("draw-dot",
                             llvm::cl::desc("Generate Graphviz .dot model visualization"),
                             llvm::cl::init(false), llvm::cl::cat(compiler_category));

fs::path prepare_dot_path(const fs::path &onnx_path) {
    fs::path dot_dir = fs::path(PROJECT_ROOT) / "data" / "dot_models";
    fs::create_directories(dot_dir);
    fs::path file = dot_dir / (onnx_path.stem().string() + ".dot");
    return file;
}

void draw_graph(const Labs::ONNX_Graph &my_graph, const fs::path &file) {
    const std::string graph_dot = Labs::build_dot(my_graph, "Arial", "#555555", 1.5);
    Labs::render_graphviz(graph_dot, file);
}

} // namespace

int main(int argc, char *argv[]) {
    Labs::init_llvm_targets();

    llvm::cl::HideUnrelatedOptions(compiler_category);
    llvm::cl::ParseCommandLineOptions(argc, argv, "ONNX Tensor Compiler\n");

    if (opt_level < 0 || opt_level > 3) {
        llvm::errs() << "Error: optimization level must be between 0 and 3 (got -O" << opt_level
                     << ")\n";
        return 1;
    }

    if (emit_action == EmitAction::Obj && output_filepath == "-") {
        llvm::errs() << "Error: output file (-o <file>) is required when using --emit=obj\n";
        return 1;
    }

    try {
        fs::path onnx_path(input_filepath.getValue());
        Labs::ONNX_Graph my_graph = Labs::importer(onnx_path);

        if (draw_dot)
            draw_graph(my_graph, prepare_dot_path(onnx_path));

        mlir::MLIRContext context;
        context.loadDialect<mlir::func::FuncDialect, mlir::arith::ArithDialect,
                            mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect>();
        mlir::OwningOpRef<mlir::ModuleOp> module = Labs::mlirGen(context, my_graph);
        Labs::lower_to_llvm(*module);

        if (emit_action == EmitAction::MLIR) {
            std::error_code ec;
            llvm::raw_fd_ostream dest(output_filepath.getValue(), ec, llvm::sys::fs::OF_Text);
            if (ec)
                throw std::runtime_error("Cannot open output file " + output_filepath + ": " +
                                         ec.message());
            module->print(dest);
            return 0;
        }

        llvm::LLVMContext llvm_context;
        std::unique_ptr<llvm::Module> llvm_module =
            Labs::translate_to_llvm_ir(module.get(), llvm_context);
        std::unique_ptr<llvm::TargetMachine> tm = Labs::create_target_machine(opt_level);
        Labs::optimize_llvm_ir(*llvm_module, *tm, opt_level);

        if (emit_action == EmitAction::LLVM) {
            std::error_code ec;
            llvm::raw_fd_ostream dest(output_filepath.getValue(), ec, llvm::sys::fs::OF_Text);
            if (ec)
                throw std::runtime_error("Cannot open output file " + output_filepath + ": " +
                                         ec.message());
            llvm_module->print(dest, nullptr);
            return 0;
        }

        auto file_type = (emit_action == EmitAction::Asm) ? llvm::CodeGenFileType::AssemblyFile
                                                          : llvm::CodeGenFileType::ObjectFile;
        Labs::emit_target_code(*llvm_module, *tm, output_filepath.getValue(), file_type);
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}