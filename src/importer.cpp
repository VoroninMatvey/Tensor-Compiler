#include "importer.hpp"
#include "graph.hpp"
#include "graph_nodes.hpp"
#include "onnx.pb.h"
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace Labs {

namespace {

void importInitializers(const ::onnx::GraphProto &onnx_graph, ONNX_Graph &my_graph) {
    for (const auto &weight : onnx_graph.initializer()) {
        DataType type;
        switch (weight.data_type()) {
        case ::onnx::TensorProto::FLOAT:
            type = DataType::Float32;
            break;
        case ::onnx::TensorProto::INT64:
            type = DataType::Int64;
            break;
        default:
            throw std::invalid_argument("Unknown type of weight");
        }

        std::vector<int64_t> shape(weight.dims().begin(), weight.dims().end());
        Value *cur_weight =
            my_graph.addValue(type, weight.name(), std::move(shape), DataStatus::Present);

        const std::string &raw_bytes = weight.raw_data();
        std::vector<std::byte> temp_vec(raw_bytes.size());
        std::memcpy(temp_vec.data(), raw_bytes.data(), raw_bytes.size());
        cur_weight->raw_data_ = std::move(temp_vec);
    }
}

} // namespace

ONNX_Graph importer(const std::string &path) {
    std::ifstream input(path, std::ios::binary);
    ::onnx::ModelProto model;

    if (!model.ParseFromIstream(std::addressof(input)))
        throw std::runtime_error("read file failure " + path);
    const ::onnx::GraphProto &onnx_graph = model.graph();
    ONNX_Graph my_graph;

    importInitializers(onnx_graph, my_graph);

    return my_graph;
}

} // namespace Labs
