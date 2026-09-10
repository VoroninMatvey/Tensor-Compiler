#include "graph.hpp"
#include "importer.hpp"
#include <iostream>
#include <string>
#include <vector>

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

void checkFile(const std::string &path) {
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
    } catch (const std::exception &e) {
        std::cout << "FAILED: " << e.what() << "\n";
    }
    std::cout << "\n";
}

int main() {
    std::vector<std::string> files = {
        "tests/specific_operations/relu.onnx", "tests/specific_operations/add.onnx",
        "tests/specific_operations/mul.onnx",  "tests/specific_operations/matmul.onnx",
        "tests/specific_operations/gemm.onnx", "tests/specific_operations/conv.onnx",
    };

    for (const auto &path : files) {
        checkFile(path);
    }

    return 0;
}