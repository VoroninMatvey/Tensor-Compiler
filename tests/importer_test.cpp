#include "importer.hpp"
#include <algorithm>
#include <filesystem>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

fs::path get_path(const std::string &file_name) {
    fs::path onnx_dir = fs::path(PROJECT_ROOT) / "data" / "specific_operations";
    return onnx_dir / file_name;
}

/*TEST(Importer, FileNotFound) {
    fs::path add_file = get_path("Unknown.onnx");
    EXPECT_THROW(Labs::importer(add_file), std::invalid_argument);
}*/

TEST(ImporterSingleNode, Add) {
    fs::path add_file = get_path("add.onnx");
    Labs::ONNX_Graph add_graph = Labs::importer(add_file);

    EXPECT_EQ(add_graph.ops_.size(), 1);
    EXPECT_EQ(add_graph.ops_[0]->op_type_, "Add");
    EXPECT_EQ(add_graph.ops_[0]->inputs_.size(), 2);
    EXPECT_EQ(add_graph.ops_[0]->outputs_.size(), 1);
    EXPECT_EQ(add_graph.ops_[0]->outputs_[0]->shape_, (std::vector<int64_t>{1, 3, 4, 4}));
    EXPECT_EQ(add_graph.outputs[0]->producer_, add_graph.ops_[0].get());
}

TEST(ImporterSingleNode, Relu) {
    fs::path add_file = get_path("relu.onnx");
    Labs::ONNX_Graph relu_graph = Labs::importer(add_file);

    EXPECT_EQ(relu_graph.ops_.size(), 1);
    EXPECT_EQ(relu_graph.ops_[0]->op_type_, "Relu");
    EXPECT_EQ(relu_graph.ops_[0]->inputs_.size(), 1);
    EXPECT_EQ(relu_graph.ops_[0]->outputs_.size(), 1);
    EXPECT_EQ(relu_graph.ops_[0]->outputs_[0]->shape_, (std::vector<int64_t>{1, 3, 4, 4}));
    EXPECT_EQ(relu_graph.outputs[0]->producer_, relu_graph.ops_[0].get());
}

TEST(ImporterSingleNode, Mul) {
    fs::path add_file = get_path("mul.onnx");
    Labs::ONNX_Graph mul_graph = Labs::importer(add_file);

    EXPECT_EQ(mul_graph.ops_.size(), 1);
    EXPECT_EQ(mul_graph.ops_[0]->op_type_, "Mul");
    EXPECT_EQ(mul_graph.ops_[0]->inputs_.size(), 2);
    EXPECT_EQ(mul_graph.ops_[0]->outputs_.size(), 1);
    EXPECT_EQ(mul_graph.ops_[0]->outputs_[0]->shape_, (std::vector<int64_t>{1, 3, 4, 4}));
    EXPECT_EQ(mul_graph.outputs[0]->producer_, mul_graph.ops_[0].get());
}

TEST(ImporterSingleNode, MatMul) {
    fs::path add_file = get_path("matmul.onnx");
    Labs::ONNX_Graph matmul_graph = Labs::importer(add_file);

    EXPECT_EQ(matmul_graph.ops_.size(), 1);
    EXPECT_EQ(matmul_graph.ops_[0]->op_type_, "MatMul");
    EXPECT_EQ(matmul_graph.ops_[0]->inputs_.size(), 2);
    EXPECT_EQ(matmul_graph.ops_[0]->outputs_.size(), 1);
    EXPECT_EQ(matmul_graph.ops_[0]->outputs_[0]->shape_, (std::vector<int64_t>{1, 4, 5}));
    EXPECT_EQ(matmul_graph.outputs[0]->producer_, matmul_graph.ops_[0].get());
}

TEST(ImporterSingleNode, Gemm) {
    fs::path add_file = get_path("gemm.onnx");
    Labs::ONNX_Graph gemm_graph = Labs::importer(add_file);

    ASSERT_EQ(gemm_graph.ops_.size(), 1);
    ASSERT_EQ(gemm_graph.ops_[0]->op_type_, "Gemm");

    Labs::Gemm &gemm = static_cast<Labs::Gemm &>(*gemm_graph.ops_[0]);

    EXPECT_EQ(gemm.inputs_.size(), 3);
    EXPECT_EQ(gemm.outputs_.size(), 1);
    EXPECT_EQ(gemm.outputs_[0]->shape_, (std::vector<int64_t>{1, 5}));
    EXPECT_EQ(gemm.outputs_[0]->producer_, &gemm);
    EXPECT_EQ(gemm.need_transA_, false);
    EXPECT_EQ(gemm.need_transB_, true);

    Labs::Value &fc_weight = *gemm_graph.name_to_value_.at("fc.weight");
    Labs::Value &fc_bias = *gemm_graph.name_to_value_.at("fc.bias");

    EXPECT_EQ(fc_weight.status_, Labs::DataStatus::Present);
    EXPECT_EQ(fc_weight.shape_, (std::vector<int64_t>{5, 10}));
    EXPECT_EQ(fc_weight.raw_data_.size(), 200);

    EXPECT_EQ(fc_bias.shape_, (std::vector<int64_t>{5}));
    EXPECT_EQ(fc_bias.raw_data_.size(), 20);
}

TEST(ImporterSingleNode, Conv) {
    fs::path add_file = get_path("conv.onnx");
    Labs::ONNX_Graph conv_graph = Labs::importer(add_file);

    ASSERT_EQ(conv_graph.ops_.size(), 1);
    ASSERT_EQ(conv_graph.ops_[0]->op_type_, "Conv");

    Labs::Conv &conv = static_cast<Labs::Conv &>(*conv_graph.ops_[0]);

    EXPECT_EQ(conv.inputs_.size(), 3);
    EXPECT_EQ(conv.outputs_.size(), 1);
    EXPECT_EQ(conv.outputs_[0]->shape_, (std::vector<int64_t>{1, 8, 8, 8}));
    EXPECT_EQ(conv.outputs_[0]->producer_, &conv);
    EXPECT_EQ(conv.pads_, (std::vector<int64_t>{1, 1, 1, 1}));
    EXPECT_EQ(conv.strides_, (std::vector<int64_t>{1, 1}));

    Labs::Value &conv_weight = *conv_graph.name_to_value_.at("conv.weight");
    Labs::Value &conv_bias = *conv_graph.name_to_value_.at("conv.bias");

    EXPECT_EQ(conv_weight.status_, Labs::DataStatus::Present);
    EXPECT_EQ(conv_weight.shape_, (std::vector<int64_t>{8, 3, 3, 3}));
    EXPECT_EQ(conv_weight.raw_data_.size(), 864);
    EXPECT_EQ(conv_bias.shape_, (std::vector<int64_t>{8}));
    EXPECT_EQ(conv_bias.raw_data_.size(), 32);
}

TEST(ImporterUniversalGraph, Sizes) {
    fs::path graph_file = get_path("universal_graph.onnx");
    Labs::ONNX_Graph big_graph = Labs::importer(graph_file);

    EXPECT_EQ(big_graph.values_.size(), 52);
    EXPECT_EQ(big_graph.ops_.size(), 28);
    EXPECT_EQ(big_graph.inputs.size(), 2);
    EXPECT_EQ(big_graph.outputs.size(), 2);
}

TEST(ImporterUniversalGraph, Inputs) {
    fs::path graph_file = get_path("universal_graph.onnx");
    Labs::ONNX_Graph big_graph = Labs::importer(graph_file);
    ASSERT_EQ(big_graph.inputs.size(), 2);

    Labs::Value &image = *big_graph.name_to_value_.at("image");
    Labs::Value &context = *big_graph.name_to_value_.at("context");

    EXPECT_EQ(big_graph.inputs[0], &image);
    EXPECT_EQ(big_graph.inputs[1], &context);

    EXPECT_EQ(image.shape_, (std::vector<int64_t>{1, 3, 16, 16}));
    EXPECT_EQ(image.producer_, nullptr);

    EXPECT_EQ(context.shape_, (std::vector<int64_t>{8, 16}));
    EXPECT_EQ(context.producer_, nullptr);
}

TEST(ImporterUniversalGraph, Outputs) {
    fs::path graph_file = get_path("universal_graph.onnx");
    Labs::ONNX_Graph big_graph = Labs::importer(graph_file);
    ASSERT_EQ(big_graph.outputs.size(), 2);

    Labs::Value &out = *big_graph.name_to_value_.at("out");
    Labs::Value &context_out = *big_graph.name_to_value_.at("context_out");

    EXPECT_EQ(big_graph.outputs[0], &out);
    EXPECT_EQ(big_graph.outputs[1], &context_out);

    EXPECT_EQ(out.shape_, (std::vector<int64_t>{1, 4, 8, 8}));
    EXPECT_EQ(out.producer_->op_type_, "Relu");

    EXPECT_EQ(context_out.shape_, (std::vector<int64_t>{8, 4}));
    EXPECT_EQ(context_out.producer_->op_type_, "Gemm");
}

TEST(ImporterUniversalGraph, MultipleConsumers) {
    fs::path graph_file = get_path("universal_graph.onnx");
    Labs::ONNX_Graph big_graph = Labs::importer(graph_file);

    Labs::Value &scaled2 = *big_graph.name_to_value_.at("scaled2");
    EXPECT_EQ(scaled2.consumers_.size(), 2);

    Labs::Value &relu_down = *big_graph.name_to_value_.at("relu_down");
    EXPECT_EQ(relu_down.consumers_.size(), 2);

    Labs::Value &fc1_relu = *big_graph.name_to_value_.at("fc1_relu");
    EXPECT_EQ(fc1_relu.consumers_.size(), 2);

    Labs::Value &scaled3 = *big_graph.name_to_value_.at("scaled3");
    EXPECT_EQ(scaled3.consumers_.size(), 1);

    Labs::Value &out = *big_graph.name_to_value_.at("out");
    EXPECT_EQ(out.consumers_.size(), 0);

    Labs::Value &context_out = *big_graph.name_to_value_.at("context_out");
    EXPECT_EQ(context_out.consumers_.size(), 0);

    Labs::Value &relu1 = *big_graph.name_to_value_.at("relu1");
    ASSERT_EQ(relu1.consumers_.size(), 2);
    EXPECT_EQ(relu1.consumers_[0]->op_type_, "Conv");
    EXPECT_EQ(relu1.consumers_[1]->op_type_, "Add");

    Labs::Value &ctx = *big_graph.name_to_value_.at("ctx");
    ASSERT_EQ(ctx.consumers_.size(), 2);
    EXPECT_EQ(ctx.consumers_[0]->op_type_, "MatMul");
    EXPECT_EQ(ctx.consumers_[1]->op_type_, "Gemm");
}

TEST(ImporterUniversalGraph, IntermediateShapes) {
    fs::path graph_file = get_path("universal_graph.onnx");
    Labs::ONNX_Graph big_graph = Labs::importer(graph_file);

    Labs::Value &conv1 = *big_graph.name_to_value_.at("conv1");
    EXPECT_EQ(conv1.shape_, (std::vector<int64_t>{1, 8, 16, 16}));

    Labs::Value &conv_dil = *big_graph.name_to_value_.at("conv_dil");
    EXPECT_EQ(conv_dil.shape_, (std::vector<int64_t>{1, 16, 8, 8}));

    Labs::Value &fc1 = *big_graph.name_to_value_.at("fc1");
    EXPECT_EQ(fc1.shape_, (std::vector<int64_t>{8, 32}));

    Labs::Value &merge_mm = *big_graph.name_to_value_.at("merge_mm");
    EXPECT_EQ(merge_mm.shape_, (std::vector<int64_t>{1, 4, 8, 8}));
}

TEST(ImporterUniversalGraph, Attributes) {
    fs::path graph_file = get_path("universal_graph.onnx");
    Labs::ONNX_Graph big_graph = Labs::importer(graph_file);

    Labs::Conv *conv_dil_prod =
        static_cast<Labs::Conv *>(big_graph.name_to_value_.at("conv_dil")->producer_);
    EXPECT_EQ(conv_dil_prod->group_, 4);

    Labs::Gemm *fc2_prod = static_cast<Labs::Gemm *>(big_graph.name_to_value_.at("fc2")->producer_);
    EXPECT_EQ(fc2_prod->alpha_, 0.5f);

    Labs::Gemm *context_out_prod =
        static_cast<Labs::Gemm *>(big_graph.name_to_value_.at("context_out")->producer_);
    EXPECT_EQ(context_out_prod->need_transA_, true);
}

TEST(ImporterUniversalGraph, Invariants) {
    fs::path graph_file = get_path("universal_graph.onnx");
    Labs::ONNX_Graph big_graph = Labs::importer(graph_file);
    std::unordered_map<const Labs::Op *, int> op_seq;

    for (int i = 0; i < big_graph.ops_.size(); ++i) {
        op_seq.insert({big_graph.ops_[i].get(), i});
    }

    for (const auto &op_uniq : big_graph.ops_) {
        const auto *op = op_uniq.get();

        for (const auto *op_input_i : op->inputs_) {
            auto it_cons =
                std::find(op_input_i->consumers_.begin(), op_input_i->consumers_.end(), op);
            EXPECT_NE(it_cons, op_input_i->consumers_.end());
        }

        for (const auto *op_output_i : op->outputs_) {
            EXPECT_EQ(op_output_i->producer_, op);
        }

        bool inputs_exist = true;
        for (const auto *op_input_i : op->inputs_) {
            if (op_input_i->producer_ == nullptr)
                continue;

            int op_num = op_seq.at(op);
            int prod_op_num = op_seq.at(op_input_i->producer_);

            if (prod_op_num >= op_num)
                inputs_exist = false;
        }
        EXPECT_EQ(inputs_exist, true);
    }
}