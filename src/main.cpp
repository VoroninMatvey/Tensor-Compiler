#include "graph.hpp"
#include "graph_builder.hpp"
#include "importer.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

void printShape(const std::vector<int64_t> &shape) {
    std::cout << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];
        if (i + 1 < shape.size())
            std::cout << ", ";
    }
    std::cout << "]";
}

void printValue(const Labs::Value *v) {
    std::cout << "  " << v->name_ << " shape=";
    printShape(v->shape_);
    std::cout << " status=" << (v->status_ == Labs::DataStatus::Present ? "Present" : "Absent");
    if (v->status_ == Labs::DataStatus::Present) {
        std::cout << " raw_data size: " << v->raw_data_.size() << " bytes";
    }
    std::cout << "\n";
}

void dumpDot(const Labs::ONNX_Graph &graph, const fs::path &out_path) {
    std::string dot = Labs::build_dot(graph, "Arial", "#555555", 1.5);

    std::ofstream file(out_path);
    if (!file) {
        std::cout << "  cannot open " << out_path << " for writing\n";
        return;
    }

    file << dot;
    std::cout << "  dot written to " << out_path << "\n";
}

void checkFile(const std::string &path, const fs::path &dot_dir) {
    std::cout << "=== " << path << " ===\n";
    try {
        Labs::ONNX_Graph graph = Labs::importer(path);

        std::cout << "All values (" << graph.values_.size() << "):\n";
        for (const auto &v : graph.values_) {
            printValue(v.get());
        }

        std::cout << "Graph inputs (" << graph.inputs.size() << "):\n";
        for (const auto *v : graph.inputs) {
            printValue(v);
        }

        std::cout << "Graph outputs (" << graph.outputs.size() << "):\n";
        for (const auto *v : graph.outputs) {
            printValue(v);
        }

        dumpDot(graph, dot_dir / (fs::path(path).stem().string() + ".dot"));
    } catch (const std::exception &e) {
        std::cout << "FAILED: " << e.what() << "\n";
    }
    std::cout << "\n";
}

int main() {
    std::vector<std::string> files = {
        "tests/specific_operations/relu.onnx",      "tests/specific_operations/add.onnx",
        "tests/specific_operations/mul.onnx",       "tests/specific_operations/matmul.onnx",
        "tests/specific_operations/gemm.onnx",      "tests/specific_operations/conv.onnx",
        "tests/specific_operations/big_graph.onnx",
    };

    fs::path dot_dir = "build/dot";
    fs::create_directories(dot_dir);

    for (const auto &path : files) {
        checkFile(path, dot_dir);
    }

    std::cout
        << "Render all: for f in build/dot/*.dot; do dot -Tpng \"$f\" -o \"${f%.dot}.png\"; done\n";

    return 0;
}