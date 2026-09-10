#include "importer.hpp"
#include "graph.hpp"
#include "graph_nodes.hpp"
#include "onnx.pb.h"
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace Labs {

namespace {

DataType define_type(int32_t elem_type, std::string_view name, std::string_view context);
std::vector<int64_t> extract_shape(const ::onnx::TensorShapeProto &shape_proto);
Value *creative_value_from_info(const ::onnx::ValueInfoProto &value_info_proto,
                                ONNX_Graph &my_graph, std::string_view context);

void importInitializers(const ::onnx::GraphProto &onnx_graph, ONNX_Graph &my_graph) {
    for (const auto &tensor_proto : onnx_graph.initializer()) {
        DataType type = define_type(tensor_proto.data_type(), tensor_proto.name(), "weight");

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
    // input
    //-------------------------------------------------------------------------
    for (const auto &value_info_proto : onnx_graph.input()) {
        if (my_graph.name_to_value_.count(value_info_proto.name()) > 0)
            throw std::runtime_error("Input tensor: " + value_info_proto.name() +
                                     " overlaps with an initializer");

        Value *cur_input = creative_value_from_info(value_info_proto, my_graph, "input");
        my_graph.inputs.push_back(cur_input);
    }

    // output
    //-------------------------------------------------------------------------
    for (const auto &value_info_proto : onnx_graph.output()) {
        Value *cur_output = creative_value_from_info(value_info_proto, my_graph, "output");
        my_graph.outputs.push_back(cur_output);
    }
}

DataType define_type(int32_t elem_type, std::string_view name, std::string_view context) {
    switch (elem_type) {
    case ::onnx::TensorProto::FLOAT:
        return DataType::Float32;
    case ::onnx::TensorProto::INT64:
        return DataType::Int64;
    default: {
        const auto &unsup_type =
            ::onnx::TensorProto_DataType_Name(static_cast<::onnx::TensorProto_DataType>(elem_type));
        throw std::runtime_error("Unsupported type of " + std::string(context) + " {" +
                                 std::string(name) + "}: " + unsup_type);
    }
    }
}

std::vector<int64_t> extract_shape(const ::onnx::TensorShapeProto &shape_proto) {
    std::vector<int64_t> shape;
    shape.reserve(shape_proto.dim_size());

    for (const auto &dim : shape_proto.dim()) {
        if (dim.has_dim_value()) {
            shape.push_back(dim.dim_value());
        } else {
            shape.push_back(-1);
        }
    }

    return shape;
}

Value *creative_value_from_info(const ::onnx::ValueInfoProto &value_info_proto,
                                ONNX_Graph &my_graph, std::string_view context) {
    if (!value_info_proto.has_type() || !value_info_proto.type().has_tensor_type())
        throw std::runtime_error(value_info_proto.name() + " isn't tensor");

    const auto &tensor_type = value_info_proto.type().tensor_type();
    DataType type = define_type(tensor_type.elem_type(), value_info_proto.name(), context);
    std::vector<int64_t> shape = extract_shape(tensor_type.shape());

    return my_graph.addValue(type, value_info_proto.name(), std::move(shape), DataStatus::Absent);
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
