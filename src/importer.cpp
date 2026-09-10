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
    for (const auto &tensor_proto : onnx_graph.initializer()) {
        DataType type;
        switch (tensor_proto.data_type()) {
        case ::onnx::TensorProto::FLOAT:
            type = DataType::Float32;
            break;
        case ::onnx::TensorProto::INT64:
            type = DataType::Int64;
            break;
        default:
            throw std::invalid_argument("Unknown type of weight");
        }

        std::vector<int64_t> shape(tensor_proto.dims().begin(), tensor_proto.dims().end());
        Value *cur_weight =
            my_graph.addValue(type, tensor_proto.name(), std::move(shape), DataStatus::Present);

        const std::string &raw_bytes = tensor_proto.raw_data();
        std::vector<std::byte> temp_vec(raw_bytes.size());
        std::memcpy(temp_vec.data(), raw_bytes.data(), raw_bytes.size());
        cur_weight->raw_data_ = std::move(temp_vec);
    }
}

void importInputOutput(const ::onnx::GraphProto &onnx_graph, ONNX_Graph &my_graph) {
    for (const auto &value_info_proto : onnx_graph.input()) {
        if (!value_info_proto.has_type() || !value_info_proto.type().has_tensor_type())
            throw std::invalid_argument(value_info_proto.name() + " isn't tensor");

        if (my_graph.name_to_value_.count(value_info_proto.name()) > 0)
            throw std::runtime_error("Input tensor: " + value_info_proto.name() +
                                     "overlaps with an initializer");

        std::vector<int64_t> shape;
        const auto &tensor_shape_proto_dim = value_info_proto.type().tensor_type().shape().dim();
        for (const auto &dim : tensor_shape_proto_dim) {
            if (dim.has_dim_value()) {
                shape.push_back(dim.dim_value());
            } else {
                shape.push_back(-1);
            }
        }

        const auto input_elem_type = value_info_proto.type().tensor_type().elem_type();
        DataType type;
        switch (input_elem_type) {
        case ::onnx::TensorProto::FLOAT:
            type = DataType::Float32;
            break;
        case ::onnx::TensorProto::INT64:
            type = DataType::Int64;
            break;
        default:
            throw std::invalid_argument(
                "Unsupported type: " +
                ::onnx::TensorProto_DataType_Name(
                    static_cast<::onnx::TensorProto_DataType>(input_elem_type)) +
                " on input");
        }
        Value *cur_input =
            my_graph.addValue(type, value_info_proto.name(), std::move(shape), DataStatus::Absent);
        my_graph.inputs.push_back(cur_input);
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
    importInputOutput(onnx_graph, my_graph);

    return my_graph;
}

} // namespace Labs
