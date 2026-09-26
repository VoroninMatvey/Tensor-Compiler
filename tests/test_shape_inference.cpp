#include "shape_inference.hpp"
#include <cstdint>
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

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
TEST(InferShape)