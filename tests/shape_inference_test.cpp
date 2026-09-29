#include "shape_inference.hpp"
#include <cstdint>
#include <gtest/gtest.h>
#include <initializer_list>
#include <stdexcept>
#include <vector>

struct Inputs {
    std::vector<Labs::Value> values_;
    std::vector<Labs::Value *> ptrs_;
};

Inputs make_input(std::initializer_list<std::vector<int64_t>> init_list) {
    Inputs input;

    for (const auto &elem : init_list) {
        Labs::Value value;
        value.shape_ = elem;
        input.values_.push_back(std::move(value));
    }

    for (auto &value : input.values_) {
        input.ptrs_.push_back(&value);
    }

    return input;
}

Labs::Gemm make_test_gemm(bool need_trans_A, bool need_trans_B) {
    Labs::Gemm gemm;
    gemm.need_transA_ = need_trans_A;
    gemm.need_transB_ = need_trans_B;

    return gemm;
}

// BROADCAST
TEST(Broadcast, UnknownVsSpecific) {
    EXPECT_EQ(Labs::broadcast(-1, 5, "Add"), 5);
    EXPECT_EQ(Labs::broadcast(6, -1, "Add"), 6);
    EXPECT_EQ(Labs::broadcast(-1, -1, "Add"), -1);
}

TEST(Broadcast, SpecificVsSpecific) {
    EXPECT_EQ(Labs::broadcast(13, 13, "Add"), 13);
    EXPECT_EQ(Labs::broadcast(1, 7, "Add"), 7);
    EXPECT_EQ(Labs::broadcast(8, 1, "Add"), 8);
}

TEST(Broadcast, IncompatibleInput) {
    EXPECT_THROW(Labs::broadcast(24, 23, "TEST_OP"), std::invalid_argument);
}

// BROADCASTSHAPES
TEST(BroadcastShapes, SpecificInputs) {
    EXPECT_EQ(Labs::broadcast_shapes({1, 3, 4, 4}, {1, 3, 4, 4}, "Add"),
              (std::vector<int64_t>{1, 3, 4, 4}));
    EXPECT_EQ(Labs::broadcast_shapes({2, 3, 4, 4}, {3, 1, 1}, "Add"),
              (std::vector<int64_t>{2, 3, 4, 4}));
    EXPECT_EQ(Labs::broadcast_shapes({4, 1}, {1, 6}, "Add"), (std::vector<int64_t>{4, 6}));
    EXPECT_EQ(Labs::broadcast_shapes({5, 10}, {10}, "Add"), (std::vector<int64_t>{5, 10}));
    EXPECT_EQ(Labs::broadcast_shapes({1, 3, 5, 7}, {}, "Add"), (std::vector<int64_t>{1, 3, 5, 7}));
    EXPECT_EQ(Labs::broadcast_shapes({}, {}, "Add"), (std::vector<int64_t>{}));
}

TEST(BroadcastShapes, UnknownInput) {
    EXPECT_EQ(Labs::broadcast_shapes({32, 2, 3, 4}, {-1, 2, 3, 4}, "Add"),
              (std::vector<int64_t>{32, 2, 3, 4}));
    EXPECT_EQ(Labs::broadcast_shapes({-1, 17}, {1, 1}, "Add"), (std::vector<int64_t>{-1, 17}));
}

TEST(BroadcastShapes, IncompatibleInput) {
    EXPECT_THROW(Labs::broadcast_shapes({1, 3, 5}, {3, 3, 3}, "Add"), std::invalid_argument);
}

TEST(BroadcastShapes, WithOffset) {
    EXPECT_EQ(Labs::broadcast_shapes({2, 4, 3}, {2, 3, 19}, "MatMul", 2),
              (std::vector<int64_t>{2}));
    EXPECT_EQ(Labs::broadcast_shapes({3}, {2, 6, 3, 5}, "MatMul", 2), (std::vector<int64_t>{2, 6}));
    EXPECT_EQ(Labs::broadcast_shapes({4, 3}, {3, 5}, "MatMul", 2), (std::vector<int64_t>{}));
}

// INFER_OP_OUTPUT_SHAPE
TEST(InferShapeRelu, CopiesInputShape) {
    Inputs inputs1 = make_input({{1, 3, 4, 4}});
    Inputs inputs2 = make_input({{}}); // scalar
    Inputs inputs3 = make_input({{-1, 3}});

    EXPECT_EQ(Labs::infer_shape_relu(inputs1.ptrs_), (std::vector<int64_t>{1, 3, 4, 4}));
    EXPECT_EQ(Labs::infer_shape_relu(inputs2.ptrs_), (std::vector<int64_t>{}));
    EXPECT_EQ(Labs::infer_shape_relu(inputs3.ptrs_), (std::vector<int64_t>{-1, 3}));
}

TEST(InferShapeAddMul, BroadcastsInputs) {
    Inputs inputs1 = make_input({{2, 3, 4, 4}, {3, 1, 1}});
    Inputs inputs2 = make_input({{1, 8}, {}});
    Inputs inputs3 = make_input({{4, 3}, {5, 3}});

    EXPECT_EQ(Labs::infer_shape_add_mul(inputs1.ptrs_, "Add"), (std::vector<int64_t>{2, 3, 4, 4}));
    EXPECT_EQ(Labs::infer_shape_add_mul(inputs2.ptrs_, "Add"), (std::vector<int64_t>{1, 8}));
    EXPECT_THROW(Labs::infer_shape_add_mul(inputs3.ptrs_, "Add"), std::invalid_argument);
}

TEST(InferShapeMatMul, Basic) {
    Inputs inputs1 = make_input({{5, 4}, {4, 7}});
    Inputs inputs2 = make_input({{2, 4, 3}, {2, 3, 5}});
    Inputs inputs3 = make_input({{2, 1, 4, 3}, {1, 5, 3, 6}});
    Inputs inputs4 = make_input({{2, 4, 3}, {3, 5}});

    EXPECT_EQ(Labs::infer_shape_matmul(inputs1.ptrs_), (std::vector<int64_t>{5, 7}));
    EXPECT_EQ(Labs::infer_shape_matmul(inputs2.ptrs_), (std::vector<int64_t>{2, 4, 5}));
    EXPECT_EQ(Labs::infer_shape_matmul(inputs3.ptrs_), (std::vector<int64_t>{2, 5, 4, 6}));
    EXPECT_EQ(Labs::infer_shape_matmul(inputs4.ptrs_), (std::vector<int64_t>{2, 4, 5}));
}

TEST(InferShapeMatMul, Vectors) {
    Inputs inputs1 = make_input({{3}, {3}});
    Inputs inputs2 = make_input({{4, 3}, {3}});
    Inputs inputs3 = make_input({{3}, {3, 5}});
    Inputs inputs4 = make_input({{3}, {2, 6, 3, 5}});

    EXPECT_EQ(Labs::infer_shape_matmul(inputs1.ptrs_), (std::vector<int64_t>{}));
    EXPECT_EQ(Labs::infer_shape_matmul(inputs2.ptrs_), (std::vector<int64_t>{4}));
    EXPECT_EQ(Labs::infer_shape_matmul(inputs3.ptrs_), (std::vector<int64_t>{5}));
    EXPECT_EQ(Labs::infer_shape_matmul(inputs4.ptrs_), (std::vector<int64_t>{2, 6, 5}));
}

TEST(InferShapeMatMul, UnknownDims) {
    Inputs inputs1 = make_input({{4, -1}, {3, 5}});
    Inputs inputs2 = make_input({{-1, 4, 3}, {3, 5}});

    EXPECT_EQ(Labs::infer_shape_matmul(inputs1.ptrs_), (std::vector<int64_t>{4, 5}));
    EXPECT_EQ(Labs::infer_shape_matmul(inputs2.ptrs_), (std::vector<int64_t>{-1, 4, 5}));
}

TEST(InferShapeMatMul, Exceptions) {
    Inputs inputs1 = make_input({{6, 7}, {5, 2}});
    Inputs inputs2 = make_input({{4, 3}, {}});
    Inputs inputs3 = make_input({{2, 4, 3}, {3, 3, 5}});

    EXPECT_THROW(Labs::infer_shape_matmul(inputs1.ptrs_), std::invalid_argument);
    EXPECT_THROW(Labs::infer_shape_matmul(inputs2.ptrs_), std::invalid_argument);
    EXPECT_THROW(Labs::infer_shape_matmul(inputs3.ptrs_), std::invalid_argument);
}

TEST(InferShapeGemm, Basic) {
    Inputs input1 = make_input({{2, 3}, {3, 4}});
    Labs::Gemm gemm1 = make_test_gemm(false, false);
    EXPECT_EQ(Labs::infer_shape_gemm(input1.ptrs_, gemm1), (std::vector<int64_t>{2, 4}));

    Inputs input2 = make_input({{3, 2}, {3, 4}});
    Labs::Gemm gemm2 = make_test_gemm(true, false);
    EXPECT_EQ(Labs::infer_shape_gemm(input2.ptrs_, gemm2), (std::vector<int64_t>{2, 4}));

    Inputs input3 = make_input({{1, 10}, {6, 10}});
    Labs::Gemm gemm3 = make_test_gemm(false, true);
    EXPECT_EQ(Labs::infer_shape_gemm(input3.ptrs_, gemm3), (std::vector<int64_t>{1, 6}));

    Inputs input4 = make_input({{5, 6}, {7, 5}});
    Labs::Gemm gemm4 = make_test_gemm(true, true);
    EXPECT_EQ(Labs::infer_shape_gemm(input4.ptrs_, gemm4), (std::vector<int64_t>{6, 7}));
}

TEST(InferShapeGemm, WithBias) {
    Inputs input1 = make_input({{2, 3}, {3, 4}, {4}});
    Labs::Gemm gemm = make_test_gemm(false, false);
    EXPECT_EQ(Labs::infer_shape_gemm(input1.ptrs_, gemm), (std::vector<int64_t>{2, 4}));

    Inputs input2 = make_input({{2, 3}, {3, 4}, {}});
    EXPECT_EQ(Labs::infer_shape_gemm(input2.ptrs_, gemm), (std::vector<int64_t>{2, 4}));

    Inputs input3 = make_input({{2, 3}, {3, 4}, {2, 1}});
    EXPECT_EQ(Labs::infer_shape_gemm(input3.ptrs_, gemm), (std::vector<int64_t>{2, 4}));

    Inputs input4 = make_input({{2, 3}, {3, 4}, {2, 4}});
    EXPECT_EQ(Labs::infer_shape_gemm(input4.ptrs_, gemm), (std::vector<int64_t>{2, 4}));
}

TEST(InferShapeGemm, Exceptions) {
    Inputs input1 = make_input({{2, 3}, {3, 4}, {3}});
    Labs::Gemm gemm = make_test_gemm(false, false);
    EXPECT_THROW(Labs::infer_shape_gemm(input1.ptrs_, gemm), std::invalid_argument);

    Inputs input2 = make_input({{2, 3}, {3, 4}, {1, 2, 4}});
    EXPECT_THROW(Labs::infer_shape_gemm(input2.ptrs_, gemm), std::invalid_argument);

    Inputs input3 = make_input({{2, 3}, {4, 5}});
    EXPECT_THROW(Labs::infer_shape_gemm(input3.ptrs_, gemm), std::invalid_argument);

    Inputs input4 = make_input({{2, 3, 4}, {4, 5}});
    EXPECT_THROW(Labs::infer_shape_gemm(input4.ptrs_, gemm), std::invalid_argument);
}

TEST(InferShapeConv, Padding) {
    Inputs input = make_input({{1, 3, 8, 8}, {8, 3, 3, 3}});
    Labs::Conv conv1, conv2, conv3;

    conv1.pads_ = {1, 1, 1, 1};
    EXPECT_EQ(Labs::infer_shape_conv(input.ptrs_, conv1), (std::vector<int64_t>{1, 8, 8, 8}));

    conv2.pads_ = {};
    EXPECT_EQ(Labs::infer_shape_conv(input.ptrs_, conv2), (std::vector<int64_t>{1, 8, 6, 6}));

    conv3.pads_ = {1, 0, 1, 0};
    EXPECT_EQ(Labs::infer_shape_conv(input.ptrs_, conv3), (std::vector<int64_t>{1, 8, 8, 6}));
}

TEST(InferShapeConv, Strides) {
    Inputs input = make_input({{1, 3, 8, 8}, {8, 3, 3, 3}});
    Labs::Conv conv1, conv2;

    conv1.strides_ = {2, 2};
    EXPECT_EQ(Labs::infer_shape_conv(input.ptrs_, conv1), (std::vector<int64_t>{1, 8, 3, 3}));

    conv2.strides_ = {2, 2};
    conv2.pads_ = {1, 1, 1, 1};
    EXPECT_EQ(Labs::infer_shape_conv(input.ptrs_, conv2), (std::vector<int64_t>{1, 8, 4, 4}));
}

TEST(InferShapeConv, Dilations) {
    Inputs input = make_input({{1, 3, 8, 8}, {8, 3, 3, 3}});
    Labs::Conv conv;
    conv.dilations_ = {2, 2};
    conv.pads_ = {2, 2, 2, 2};
    EXPECT_EQ(Labs::infer_shape_conv(input.ptrs_, conv), (std::vector<int64_t>{1, 8, 8, 8}));
}

TEST(InferShapeConv, Groups) {
    Inputs input = make_input({{1, 16, 8, 8}, {16, 4, 3, 3}});
    Labs::Conv conv;
    conv.group_ = 4;
    conv.pads_ = {1, 1, 1, 1};

    EXPECT_EQ(Labs::infer_shape_conv(input.ptrs_, conv), (std::vector<int64_t>{1, 16, 8, 8}));
}

TEST(InferShapeConv, DefaultAttributes) {
    Inputs input = make_input({{1, 3, 8, 8}, {8, 3, 3, 3}});
    Labs::Conv conv;
    Labs::infer_shape_conv(input.ptrs_, conv);

    EXPECT_EQ(conv.kernel_shape_, (std::vector<int64_t>{3, 3}));
    EXPECT_EQ(conv.strides_, (std::vector<int64_t>{1, 1}));
    EXPECT_EQ(conv.dilations_, (std::vector<int64_t>{1, 1}));
    EXPECT_EQ(conv.pads_, (std::vector<int64_t>{0, 0, 0, 0}));
}

TEST(InferShapeConv, WithBias) {
    Inputs input1 = make_input({{1, 3, 8, 8}, {8, 3, 3, 3}, {8}});
    Labs::Conv conv1, conv2;
    EXPECT_EQ(Labs::infer_shape_conv(input1.ptrs_, conv1), (std::vector<int64_t>{1, 8, 6, 6}));

    Inputs input2 = make_input({{1, 3, 8, 8}, {8, 3, 3, 3}, {7}});
    EXPECT_THROW(Labs::infer_shape_conv(input2.ptrs_, conv2), std::invalid_argument);
}

TEST(InferShapeConv, Exceptions) {
    Inputs input1 = make_input({{1, 3, 8, 8}, {8, 3, 3, 3}});
    Labs::Conv conv1, conv2, conv3;
    conv1.auto_pad_ = "SAME_UPPER";
    EXPECT_THROW(Labs::infer_shape_conv(input1.ptrs_, conv1), std::invalid_argument);

    Inputs input2 = make_input({{1, 3, 2, 2}, {8, 3, 3, 3}});
    EXPECT_THROW(Labs::infer_shape_conv(input2.ptrs_, conv2), std::invalid_argument);

    Inputs input3 = make_input({{1, 4, 8, 8}, {8, 3, 3, 3}});
    EXPECT_THROW(Labs::infer_shape_conv(input3.ptrs_, conv3), std::invalid_argument);
}