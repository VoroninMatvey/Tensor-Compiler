#include "shape_inference.hpp"
#include "graph_nodes.hpp"
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace Labs {

namespace {

void read_gemm_input_dims(int64_t &first, int64_t &second, const std::vector<int64_t> &shape,
                          bool need_trans);

int64_t compute_spatial_conv_dim(const Conv &conv, const std::vector<int64_t> &shapeX, int iter);
void check_conv_channels(const Conv &conv, int64_t C_X, int64_t C_div_group_W);
void check_conv_bias(const std::vector<Value *> &input);
void check_and_fill_conv_field(Conv &conv, const std::vector<int64_t> &shapeX,
                               const std::vector<int64_t> &shapeW);

} // namespace

int64_t broadcast(int64_t num1, int64_t num2, std::string_view op_type) {
    if (num1 == num2) {
        return num1;
    } else if (num1 == 1) {
        return num2;
    } else if (num2 == 1) {
        return num1;
    } else if (num1 == -1) {
        return num2;
    } else if (num2 == -1) {
        return num1;
    }

    throw std::invalid_argument("invalid argument of operation: " + std::string(op_type));
}

std::vector<int64_t> broadcast_shapes(const std::vector<int64_t> &shape1,
                                      const std::vector<int64_t> &shape2, std::string_view op_type,
                                      int offset) {
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

// Relu
std::vector<int64_t> infer_shape_relu(const std::vector<Value *> &input) {
    return input[0]->shape_;
}

// Add and Mul
std::vector<int64_t> infer_shape_add_mul(const std::vector<Value *> &input,
                                         std::string_view op_type) {
    const std::vector<int64_t> &shape1 = input[0]->shape_;
    const std::vector<int64_t> &shape2 = input[1]->shape_;

    return broadcast_shapes(shape1, shape2, op_type);
}

// MatMul
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

    std::vector<int64_t> op_shape = broadcast_shapes(shapeA, shapeB, "MatMul", 2);

    if (!A_was_vector)
        op_shape.push_back(M);
    if (!B_was_vector)
        op_shape.push_back(N);

    return op_shape;
}

// Gemm
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

// Conv
std::vector<int64_t> infer_shape_conv(const std::vector<Value *> &input, Conv &conv) {
    const std::vector<int64_t> &shapeX = input[0]->shape_;
    const std::vector<int64_t> &shapeW = input[1]->shape_;

    if (shapeX.size() < 3)
        throw std::invalid_argument("Conv: input rank must be at least 3");

    if (shapeX.size() != shapeW.size())
        throw std::invalid_argument("Conv: inputs has different ranks");

    check_and_fill_conv_field(conv, shapeX, shapeW);
    check_conv_channels(conv, shapeX[1], shapeW[1]);

    int dim = shapeX.size() - 2;
    std::vector<int64_t> shape_out = {shapeX[0], shapeW[0]};
    for (int i = 0; i < dim; ++i) {
        int64_t out_i = compute_spatial_conv_dim(conv, shapeX, i);
        shape_out.push_back(out_i);
    }

    check_conv_bias(input);
    return shape_out;
}

namespace {
// Gemm
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

// Conv
void check_and_fill_conv_field(Conv &conv, const std::vector<int64_t> &shapeX,
                               const std::vector<int64_t> &shapeW) {
    int dim = shapeX.size() - 2;

    if (conv.kernel_shape_.empty()) {
        conv.kernel_shape_.assign(shapeW.begin() + 2, shapeW.end());
    } else {
        if (conv.kernel_shape_.size() != dim)
            throw std::invalid_argument("Conv: erroneous reading of shape_kernel");
    }

    if (conv.strides_.empty()) {
        conv.strides_.assign(dim, 1);
    } else {
        if (conv.strides_.size() != dim)
            throw std::invalid_argument("Conv: erroneous reading of strides");
    }

    if (conv.dilations_.empty()) {
        conv.dilations_.assign(dim, 1);
    } else {
        if (conv.dilations_.size() != dim)
            throw std::invalid_argument("Conv: erroneous reading of dilations");
    }

    // Then come back here and finish it
    if (conv.auto_pad_ != "NOTSET")
        throw std::invalid_argument("This type of auto_pad is not supported: " + conv.auto_pad_);

    if (conv.pads_.empty()) {
        conv.pads_.assign(2 * dim, 0);
    } else {
        if (conv.pads_.size() != 2 * dim)
            throw std::invalid_argument("Conv: erroneous reading of pads");
    }
}

void check_conv_channels(const Conv &conv, int64_t C_X, int64_t C_div_group_W) {
    if (conv.group_ <= 0) {
        throw std::invalid_argument("Conv: group less or equal 0");
    }

    if (C_X != -1) {
        if (C_X % conv.group_ != 0)
            throw std::invalid_argument("Conv: C not divisible by group");

        if (C_div_group_W != -1 && C_div_group_W != C_X / conv.group_)
            throw std::invalid_argument("Conv: channel mismatch");
    }
}

int64_t compute_spatial_conv_dim(const Conv &conv, const std::vector<int64_t> &shapeX, int iter) {
    int64_t in = shapeX[iter + 2];
    if (in == -1) {
        return -1;
    }

    if (conv.strides_[iter] <= 0)
        throw std::invalid_argument("Conv: Invalid value of strides[" + std::to_string(iter) + "]");

    if (conv.dilations_[iter] <= 0)
        throw std::invalid_argument("Conv: Invalid value of dilations[" + std::to_string(iter) +
                                    "]");

    int dim = shapeX.size() - 2;
    int64_t field = in + conv.pads_[iter] + conv.pads_[iter + dim];
    int64_t window = (conv.kernel_shape_[iter] - 1) * conv.dilations_[iter] + 1;

    if (field - window < 0)
        throw std::invalid_argument("Conv: kernel larger than padded input");

    int64_t out = (field - window) / conv.strides_[iter] + 1;
    return out;
}

void check_conv_bias(const std::vector<Value *> &input) {
    if (input.size() == 3) {
        const std::vector<int64_t> &bias = input[2]->shape_;
        const std::vector<int64_t> &shapeW = input[1]->shape_;

        if (bias.size() != 1) {
            throw std::invalid_argument("Conv: bias must have rank 1");
        }

        if (bias[0] != -1 && shapeW[0] != -1 && shapeW[0] != bias[0])
            throw std::invalid_argument("Conv: B size must match number of filters");
    }
}

} // namespace

} // namespace Labs