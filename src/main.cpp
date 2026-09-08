#include "graph.hpp"
#include "importer.hpp"
#include <iostream>

int main() {
    Labs::ONNX_Graph graph = Labs::importer("tests/specific_operations/conv.onnx");
    std::cout << "Values (weights): " << graph.values_.size() << "\n";
    for (const auto &v : graph.values_) {
        std::cout << "  " << v->name_ << " raw_data size: " << v->raw_data_.size() << " bytes\n";
    }
    return 0;
}