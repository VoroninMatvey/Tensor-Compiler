#include "importer.hpp"
#include "graph.hpp"
#include "graph_nodes.hpp"
#include "onnx.pb.h"
#include "shape_inference.hpp"
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace Labs {

namespace {

DataType convert_onnx_type(int32_t elem_type, std::string_view name, std::string_view context);
std::vector<int64_t> extract_shape(const ::onnx::TensorShapeProto &shape_proto);
Value *create_value_from_value_info(const ::onnx::ValueInfoProto &value_info_proto,
                                    ONNX_Graph &my_graph, std::string_view context);
Op *create_op(const ::onnx::NodeProto &node_proto, std::vector<Value *> input, ONNX_Graph &my_graph,
              std::vector<int64_t> &out_shape);

void fill_gemm_attr(const ::onnx::NodeProto &node_proto, Gemm &gemm);
void fill_conv_attr(const ::onnx::NodeProto &node_proto, Conv &conv);

// importInitializers
void importInitializers(const ::onnx::GraphProto &onnx_graph, ONNX_Graph &my_graph) {
    for (const auto &tensor_proto : onnx_graph.initializer()) {
        DataType type = convert_onnx_type(tensor_proto.data_type(), tensor_proto.name(), "weight");

        std::vector<int64_t> shape(tensor_proto.dims().begin(), tensor_proto.dims().end());
        Value *cur_weight =
            my_graph.addValue(type, tensor_proto.name(), std::move(shape), DataStatus::Present);

        const std::string &raw_bytes = tensor_proto.raw_data();
        if (raw_bytes.empty())
            throw std::runtime_error("Initializer " + tensor_proto.name() +
                                     ": data not found in raw_data (unsupported storage format)");

        std::vector<std::byte> temp_vec(raw_bytes.size());
        std::memcpy(temp_vec.data(), raw_bytes.data(), raw_bytes.size());
        cur_weight->raw_data_ = std::move(temp_vec);
    }
}

DataType convert_onnx_type(int32_t elem_type, std::string_view name, std::string_view context) {
    switch (elem_type) {
    case ::onnx::TensorProto::FLOAT:
        return DataType::Float32;
    case ::onnx::TensorProto::INT64:
        return DataType::Int64;
    default: {
        const auto &unsupp_type =
            ::onnx::TensorProto_DataType_Name(static_cast<::onnx::TensorProto_DataType>(elem_type));
        throw std::runtime_error("Unsupported type of " + std::string(context) + " {" +
                                 std::string(name) + "}: " + unsupp_type);
    }
    }
}

// importInputOutput
void importInputOutput(const ::onnx::GraphProto &onnx_graph, ONNX_Graph &my_graph) {
    for (const auto &value_info_proto : onnx_graph.input()) {
        if (my_graph.name_to_value_.count(value_info_proto.name()) > 0)
            throw std::runtime_error("Input tensor: " + value_info_proto.name() +
                                     " overlaps with an initializer");

        Value *cur_input = create_value_from_value_info(value_info_proto, my_graph, "input");
        my_graph.inputs.push_back(cur_input);
    }

    for (const auto &value_info_proto : onnx_graph.output()) {
        Value *cur_output = create_value_from_value_info(value_info_proto, my_graph, "output");
        my_graph.outputs.push_back(cur_output);
    }
}

Value *create_value_from_value_info(const ::onnx::ValueInfoProto &value_info_proto,
                                    ONNX_Graph &my_graph, std::string_view context) {
    if (!value_info_proto.has_type() || !value_info_proto.type().has_tensor_type())
        throw std::runtime_error(value_info_proto.name() + " isn't tensor");

    const auto &tensor_type = value_info_proto.type().tensor_type();
    DataType type = convert_onnx_type(tensor_type.elem_type(), value_info_proto.name(), context);
    std::vector<int64_t> shape = extract_shape(tensor_type.shape());

    return my_graph.addValue(type, value_info_proto.name(), std::move(shape), DataStatus::Absent);
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

// importNodes
void importNodes(const ::onnx::GraphProto &onnx_graph, ONNX_Graph &my_graph) {
    for (const auto &node_proto : onnx_graph.node()) {
        const std::string &op_type = node_proto.op_type();
        std::vector<Value *> input;

        for (const auto &input_elem : node_proto.input()) {
            auto it = my_graph.name_to_value_.find(input_elem);
            if (it == my_graph.name_to_value_.end())
                throw std::runtime_error("Error during operator traversal, node: " + op_type);

            input.push_back(it->second);
        }

        std::vector<int64_t> out_shape;
        Op *op_ptr = create_op(node_proto, std::move(input), my_graph, out_shape);

        for (auto &input_elem : op_ptr->inputs_) {
            input_elem->consumers_.push_back(op_ptr);
        }

        DataType type = op_ptr->inputs_[0]->type_; // only for my specific 6 operators
        const std::string &tensor_name = node_proto.output()[0];

        auto it_out = my_graph.name_to_value_.find(tensor_name);
        Value *val_out_ptr = nullptr;
        if (it_out != my_graph.name_to_value_.end()) {
            val_out_ptr = it_out->second;

            if (val_out_ptr->producer_ != nullptr)
                throw std::runtime_error("Value " + tensor_name + " is produced twice");
        } else {
            val_out_ptr = my_graph.addValue(type, tensor_name, out_shape, DataStatus::Absent);
        }

        val_out_ptr->producer_ = op_ptr;
        op_ptr->outputs_.push_back(val_out_ptr);
    }
}

Op *create_op(const ::onnx::NodeProto &node_proto, std::vector<Value *> input, ONNX_Graph &my_graph,
              std::vector<int64_t> &out_shape) {
    const std::string &op_type = node_proto.op_type();
    Op *op_ptr = nullptr;

    if (op_type == "Add") {
        op_ptr = my_graph.addOp<Add>();
        op_ptr->inputs_ = std::move(input);
        out_shape = infer_shape_add_mul(op_ptr->inputs_, "Add");
    } else if (op_type == "Mul") {
        op_ptr = my_graph.addOp<Mul>();
        op_ptr->inputs_ = std::move(input);
        out_shape = infer_shape_add_mul(op_ptr->inputs_, "Mul");
    } else if (op_type == "Relu") {
        op_ptr = my_graph.addOp<Relu>();
        op_ptr->inputs_ = std::move(input);
        out_shape = infer_shape_relu(op_ptr->inputs_);
    } else if (op_type == "MatMul") {
        op_ptr = my_graph.addOp<MatMul>();
        op_ptr->inputs_ = std::move(input);
        out_shape = infer_shape_matmul(op_ptr->inputs_);
    } else if (op_type == "Gemm") {
        Gemm *gemm_ptr = my_graph.addOp<Gemm>();
        gemm_ptr->inputs_ = std::move(input);
        fill_gemm_attr(node_proto, *gemm_ptr);
        out_shape = infer_shape_gemm(gemm_ptr->inputs_, *gemm_ptr);
        op_ptr = gemm_ptr;
    } else if (op_type == "Conv") {
        Conv *conv_ptr = my_graph.addOp<Conv>();
        conv_ptr->inputs_ = std::move(input);
        fill_conv_attr(node_proto, *conv_ptr);
        out_shape = infer_shape_conv(conv_ptr->inputs_, *conv_ptr);
        op_ptr = conv_ptr;
    } else {
        throw std::runtime_error("Unsupported type: " + op_type);
    }

    return op_ptr;
}

void fill_gemm_attr(const ::onnx::NodeProto &node_proto, Gemm &gemm) {
    for (const auto &node_attr : node_proto.attribute()) {
        const std::string &attr_name = node_attr.name();
        if (attr_name == "alpha") {
            gemm.alpha_ = node_attr.f();
        } else if (attr_name == "beta") {
            gemm.beta_ = node_attr.f();
        } else if (attr_name == "transA") {
            gemm.need_transA_ = (0 != node_attr.i());
        } else if (attr_name == "transB") {
            gemm.need_transB_ = (0 != node_attr.i());
        }
    }
}

void fill_conv_attr(const ::onnx::NodeProto &node_proto, Conv &conv) {
    for (const auto &node_attr : node_proto.attribute()) {
        const std::string &attr_name = node_attr.name();

        if (attr_name == "dilations") {
            conv.dilations_.assign(node_attr.ints().begin(), node_attr.ints().end());
        } else if (attr_name == "kernel_shape") {
            conv.kernel_shape_.assign(node_attr.ints().begin(), node_attr.ints().end());
        } else if (attr_name == "pads") {
            conv.pads_.assign(node_attr.ints().begin(), node_attr.ints().end());
        } else if (attr_name == "strides") {
            conv.strides_.assign(node_attr.ints().begin(), node_attr.ints().end());
        } else if (attr_name == "auto_pad") {
            conv.auto_pad_ = node_attr.s();
        } else if (attr_name == "group") {
            conv.group_ = node_attr.i();
        }
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
    importNodes(onnx_graph, my_graph);

    return my_graph;
}

} // namespace Labs
