#include "backend/codegen.hpp"
#include "backend/lowering.hpp"
#include "backend/mlir_gen.hpp"
#include "frontend/graph.hpp"
#include "frontend/graph_builder.hpp"
#include "frontend/importer.hpp"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Target/TargetMachine.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

namespace fs = std::filesystem;

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

bool has_draw_flag(int argc, char *argv[]) {
    bool draw = false;

    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--draw-dot") == 0)
            draw = true;
    }
    return draw;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <something.onnx> [--draw-dot]\n";
        return 1;
    }

    try {
        Labs::ONNX_Graph my_graph = Labs::importer(argv[1]);

        //----------------------------
        mlir::MLIRContext context;
        context.loadDialect<mlir::func::FuncDialect, mlir::arith::ArithDialect,
                            mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect>();
        mlir::OwningOpRef<mlir::ModuleOp> module = Labs::mlirGen(context, my_graph);
        Labs::lower_to_llvm(*module);
        // module->dump();
        // std::cout << std::endl;

        llvm::LLVMContext llvm_context;
        std::unique_ptr<llvm::Module> llvm_module =
            Labs::translate_to_llvm_ir(module.get(), llvm_context);
        Labs::init_llvm_targets();
        std::unique_ptr<llvm::TargetMachine> tm = Labs::create_target_machine();
        Labs::optimize_llvm_ir(*llvm_module, tm.get(), 3);
        llvm_module->print(llvm::outs(), nullptr);
        //----------------------------

        if (has_draw_flag(argc, argv)) {
            draw_graph(my_graph, prepare_dot_path(argv[1]));
        }
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}