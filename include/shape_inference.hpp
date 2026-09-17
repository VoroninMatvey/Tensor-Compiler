#pragma once
#include "graph_nodes.hpp"
#include <cstdint>
#include <string_view>
#include <vector>

namespace Labs {

int64_t broadcast(int64_t num1, int64_t num2, std::string_view op_type);
std::vector<int64_t> broadcast_shapes(const std::vector<int64_t> &shape1,
                                      const std::vector<int64_t> &shape2, std::string_view op_type,
                                      int offset = 0);

std::vector<int64_t> infer_shape_relu(const std::vector<Value *> &input);
std::vector<int64_t> infer_shape_matmul(const std::vector<Value *> &input);
std::vector<int64_t> infer_shape_conv(const std::vector<Value *> &input, Conv &conv);
std::vector<int64_t> infer_shape_gemm(const std::vector<Value *> &input, const Gemm &gemm);
std::vector<int64_t> infer_shape_add_mul(const std::vector<Value *> &input,
                                         std::string_view op_type);
} // namespace Labs