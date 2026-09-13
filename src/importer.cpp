#include "importer.hpp"
#include "graph.hpp"
#include "graph_nodes.hpp"
#include "onnx.pb.h"
#include <algorithm>
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
Value *create_value_from_value_info(const ::onnx::ValueInfoProto &value_info_proto,
                                    ONNX_Graph &my_graph, std::string_view context);

void importInitializers(const ::onnx::GraphProto &onnx_graph, ONNX_Graph &my_graph) {
    for (const auto &tensor_proto : onnx_graph.initializer()) {
        DataType type = define_type(tensor_proto.data_type(), tensor_proto.name(), "weight");

        std::vector<int64_t> shape(tensor_proto.dims().begin(), tensor_proto.dims().end());
        Value *cur_weight =
            my_graph.addValue(type, tensor_proto.name(), std::move(shape), DataStatus::Present);

        // maybe change
        const std::string &raw_bytes = tensor_proto.raw_data();
        std::vector<std::byte> temp_vec(raw_bytes.size());
        std::memcpy(temp_vec.data(), raw_bytes.data(), raw_bytes.size());
        cur_weight->raw_data_ = std::move(temp_vec);
    }
}

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

void importNodes(const ::onnx::GraphProto &onnx_graph, ONNX_Graph &my_graph) {
    for (const auto &node_proto : onnx_graph.node()) {
        const std::string op_type = node_proto.op_type();
        std::vector<Value *> input;

        for (const auto &input_elem : node_proto.input()) {
            auto it = my_graph.name_to_value_.find(input_elem);
            if (it == my_graph.name_to_value_.end())
                throw std::runtime_error("Error during operator traversal, node: " + op_type);

            input.push_back(it->second);
        }

        Op *op_ptr = nullptr;
        std::vector<int64_t> out_shape;

        if (op_type == "Add") {
            op_ptr = my_graph.addOp<Add>();
        } else if (op_type == "Mul") {
            op_ptr = my_graph.addOp<Mul>();
        } else if (op_type == "Relu") {
            op_ptr = my_graph.addOp<Relu>();
        } else if (op_type == "MatMul") {
            op_ptr = my_graph.addOp<MatMul>();
        } else if (op_type == "Gemm") {
            op_ptr = my_graph.addOp<Gemm>();
        } else if (op_type == "Conv") {
            op_ptr = my_graph.addOp<Conv>();
        } else {
            throw std::runtime_error("Unsupported type: " + op_type);
        }

        /*for (const auto &output_elem : node_proto.output()) {
            auto it = my_graph.name_to_value_.find(output_elem);
            if (it != my_graph.name_to_value_.end()) {
                op_ptr->outputs_.push_back(it->second);
                it->second->producer_ = op_ptr;
            }
        }*/
    }
}

void print_attr(const ::onnx::GraphProto &onnx_graph, ONNX_Graph &my_graph) {
    for (const auto &node_proto : onnx_graph.node()) {
        for (const auto &node_attr : node_proto.attribute()) {
            std::cout << "------------------------------------------------------" << std::endl;
            std::cout << node_attr.name() << std::endl;
            std::cout << "------------------------------------------------------" << std::endl;
        }
    }
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

int64_t broadcast(int64_t num1, int64_t num2, std::string_view op_type) {
    if (num1 == num2) {
        return num1;
    } else if (num1 == 1) {
        return num2;
    } else if (num2 == 1) {
        return num1;
    } else if (num1 == -1 || num2 == -1) {
        return -1;
    }

    throw std::invalid_argument("invalid argument of operation: " + std::string(op_type));
}

std::vector<int64_t> broadcast_shapes(const std::vector<int64_t> &shape1,
                                      const std::vector<int64_t> &shape2, std::string_view op_type,
                                      int offset = 0) {
    int rank1 = shape1.size() - offset;
    int rank2 = shape2.size() - offset;
    int out_rank = std::max(rank1, rank2);
    std::vector<int64_t> op_shape(out_rank);

    for (int i = 0; i < out_rank; ++i) {
        int64_t num1 = (i < rank1) ? shape1[rank1 - 1 - i] : 1;
        int64_t num2 = (i < rank2) ? shape2[rank2 - 1 - i] : 1;

        op_shape[out_rank - 1 - i] = broadcast(num1, num2, op_type);
    }

    return op_shape;
}

std::vector<int64_t> infer_shape_relu(const std::vector<Value *> &input) {
    return input[0]->shape_;
}

std::vector<int64_t> infer_shape_add_mul(const std::vector<Value *> &input,
                                         std::string_view op_type) {
    const std::vector<int64_t> &shape1 = input[0]->shape_;
    const std::vector<int64_t> &shape2 = input[1]->shape_;

    return broadcast_shapes(shape1, shape2, op_type);
}

std::vector<int64_t> infer_shape_matmul(const std::vector<Value *> &input) {
    std::vector<int64_t> shapeA = input[0]->shape_;
    std::vector<int64_t> shapeB = input[1]->shape_;

    if (shapeA.empty() || shapeB.empty())
        throw std::invalid_argument("MatMul does not accept scalar inputs");

    bool A_was_vector = (shapeA.size() == 1);
    if (A_was_vector)
        shapeA.insert(shapeA.begin(), 1);

    bool B_was_vector = (shapeB.size() == 1);
    if (B_was_vector)
        shapeB.push_back(1);

    int rankA = shapeA.size();
    int rankB = shapeB.size();

    int64_t M = shapeA[rankA - 2];
    int64_t N = shapeB[rankB - 1];
    int64_t K_A = shapeA[rankA - 1];
    int64_t K_B = shapeB[rankB - 2];

    if (K_A != K_B && K_A != -1 && K_B != -1)
        throw std::invalid_argument("MatMul: incompatible inner dimensions " + std::to_string(K_A) +
                                    " and " + std::to_string(K_B));

    int batch_rank = std::max(rankA, rankB) - 2;
    std::vector<int64_t> op_shape = broadcast_shapes(shapeA, shapeB, "MatMul", 2);

    if (!A_was_vector)
        op_shape.push_back(M);
    if (!B_was_vector)
        op_shape.push_back(N);

    return op_shape;
}

void read_gemm_input_dims(int64_t &first, int64_t &second, const std::vector<int64_t> &shape,
                          bool need_trans) {
    if (!need_trans) {
        first = shape[0];
        second = shape[1];
    } else {
        first = shape[1];
        second = shape[0];
    }
}

std::vector<int64_t> infer_shape_gemm(const std::vector<Value *> &input, const Gemm &gemm) {
    const std::vector<int64_t> &shapeA = input[0]->shape_;
    const std::vector<int64_t> &shapeB = input[1]->shape_;

    if (shapeA.size() != 2 || shapeB.size() != 2)
        throw std::invalid_argument("Invalid input form for gemm operator");

    int64_t M, K_A, K_B, N;
    read_gemm_input_dims(M, K_A, shapeA, gemm.need_transA_);
    read_gemm_input_dims(K_B, N, shapeB, gemm.need_transB_);

    if (K_A != K_B && K_A != -1 && K_B != -1)
        throw std::invalid_argument("Gemm: incompatible inner dimensions " + std::to_string(K_A) +
                                    " and " + std::to_string(K_B));

    if (input.size() == 3) {
        const std::vector<int64_t> &shapeC = input[2]->shape_;
        if (shapeC.size() > 2)
            throw std::invalid_argument("Gemm: C has rank greater than 2");

        broadcast_shapes({M, N}, shapeC, "Gemm"); // to check compatibility
    }

    return {M, N};
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

Value *create_value_from_value_info(const ::onnx::ValueInfoProto &value_info_proto,
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
    print_attr(onnx_graph, my_graph);

    return my_graph;
}

} // namespace Labs
