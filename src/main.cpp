#include "graph.hpp"
#include "graph_nodes.hpp"
#include <iostream>

int main() {
    Labs::ONNX_Graph graph;
    Labs::Value *input =
        graph.addValue(Labs::data_type::Float32, "input", {1, 10}, Labs::data_status::Absent);
    Labs::Value *weight =
        graph.addValue(Labs::data_type::Float32, "weight", {10, 10}, Labs::data_status::Present);
    Labs::Value *matmulOut =
        graph.addValue(Labs::data_type::Float32, "matmul_out", {1, 10}, Labs::data_status::Absent);

    Labs::MatMul *mm = graph.addOp<Labs::MatMul>();
    mm->inputs_ = {input, weight};
    mm->outputs_ = {matmulOut};

    std::cout << "values: " << graph.values_.size() << ", ops: " << graph.ops_.size() << "\n";
    return 0;
}