#include "mlir_gen.hpp"

#include "graph.hpp"
#include "graph_nodes.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <functional>
#include <ranges>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// After that Value - mlir::Value but Labs::Value - Labs::Value
using namespace mlir;

namespace {
using SplitResult = std::pair<std::vector<int64_t>, SmallVector<ReassociationIndices>>;
using BodyBuilder = std::function<void(OpBuilder &, Location, ValueRange)>;

SmallVector<int64_t> to_mlir_shape(const std::vector<int64_t> &val_shape);
SplitResult split_dim(ArrayRef<int64_t> shape, int64_t num_groups, std::size_t pos);
void check_supported_type(Type elem_type);
void check_static_shapes(const Labs::Op *op);
bool is_dynamic(const Labs::Value *val);

Value create_add(OpBuilder &b, Location loc, Value x, Value y);
Value create_mul(OpBuilder &b, Location loc, Value x, Value y);
Value create_max(OpBuilder &b, Location loc, Value x, Value y);
Value create_generic(OpBuilder &b, Location loc, ValueRange inputs, Value init,
                     ArrayRef<AffineMap> maps, ArrayRef<utils::IteratorType> iters,
                     const BodyBuilder &body);

SmallVector<utils::IteratorType> make_iters(std::size_t n_par, std::size_t n_red);
AffineMap make_map(OpBuilder &b, std::size_t loops_amount, ArrayRef<AffineExpr> exprs);
Value make_empty_tensor(OpBuilder &b, Location loc, RankedTensorType out_type);
Value make_zero_tensor(OpBuilder &b, Location loc, RankedTensorType out_type);

class MLIRGenImpl final {

  public:
    explicit MLIRGenImpl(MLIRContext &context) : builder_(&context) {}

    OwningOpRef<ModuleOp> mlirGen(const Labs::ONNX_Graph &my_graph) {
        OwningOpRef owner = ModuleOp::create(builder_.getUnknownLoc());
        theModule_ = owner.get();

        func::FuncOp main = create_main(my_graph.inputs, my_graph.outputs);
        builder_.setInsertionPointToStart(&main.front());
        gen_weights(my_graph.values_);

        for (const auto &op_ref : my_graph.ops_) {
            Labs::Op *op = op_ref.get();
            check_static_shapes(op);

            const std::string &op_type = op->op_type_;
            if (op_type == "Relu") {
                gen_relu(*op);
            } else if (op_type == "Add") {
                gen_add(*op);
            } else if (op_type == "Mul") {
                gen_mul(*op);
            } else if (op_type == "MatMul") {
                gen_matmul(*op);
            } else if (op_type == "Gemm") {
                gen_gemm(*op);
            } else if (op_type == "Conv") {
                gen_conv(*op);
            } else {
                throw std::runtime_error("unsupported gen operation now: " + op_type);
            }
        }

        SmallVector<Value> results;
        for (const auto *out_val : my_graph.outputs) {
            results.push_back(value_table_.at(out_val));
        }
        func::ReturnOp::create(builder_, builder_.getUnknownLoc(), results);

        return owner;
    }

  private:
    func::FuncOp create_main(const std::vector<Labs::Value *> &inputs,
                             const std::vector<Labs::Value *> &outputs) {
        Location location = make_loc("onnx_graph");

        SmallVector<Type> argsTypes = get_tensor_types(inputs);
        SmallVector<Type> resTypes = get_tensor_types(outputs);

        FunctionType funcType = builder_.getFunctionType(argsTypes, resTypes);
        builder_.setInsertionPointToStart(theModule_.getBody());
        func::FuncOp main = func::FuncOp::create(builder_, location, "main", funcType);

        Block *entry = main.addEntryBlock();

        assert(inputs.size() == entry->getNumArguments());
        for (std::size_t i = 0; i < inputs.size(); ++i)
            insert_into_value_table(inputs[i], entry->getArgument(i));

        return main;
    }

  private:
    ModuleOp theModule_;
    OpBuilder builder_;
    std::unordered_map<const Labs::Value *, Value> value_table_;

    RankedTensorType get_tensor_type(const Labs::Value *value) {
        Type mlir_type;
        switch (value->type_) {
        case Labs::DataType::Float32:
            mlir_type = builder_.getF32Type();
            break;
        case Labs::DataType::Int64:
            mlir_type = builder_.getI64Type();
            break;
        default:
            throw std::runtime_error("Unsupported type");
        }

        SmallVector<int64_t> mlir_shape = to_mlir_shape(value->shape_);
        return RankedTensorType::get(mlir_shape, mlir_type);
    }

    SmallVector<Type> get_tensor_types(const std::vector<Labs::Value *> &vec) {
        SmallVector<Type> vecTypes;
        vecTypes.reserve(vec.size());

        for (const auto *value : vec) {
            vecTypes.push_back(get_tensor_type(value));
        }
        return vecTypes;
    }

    void gen_weights(const std::vector<std::unique_ptr<Labs::Value>> &values) {
        auto is_weight = [](const auto &labs_value) {
            return labs_value->status_ == Labs::DataStatus::Present;
        };

        for (const auto &uniq_value : values | std::views::filter(is_weight))
            gen_weight(uniq_value.get());
    }

    void gen_weight(const Labs::Value *value_ptr) {
        RankedTensorType type = get_tensor_type(value_ptr);
        const char *raw_array = reinterpret_cast<const char *>(value_ptr->raw_data_.data());
        ArrayRef<char> bytes(raw_array, value_ptr->raw_data_.size());
        DenseElementsAttr attr = DenseElementsAttr::getFromRawBuffer(type, bytes);

        Location location = make_loc("main");
        Value cnst = arith::ConstantOp::create(builder_, location, attr);
        insert_into_value_table(value_ptr, cnst);
    }

    Location make_loc(const std::string &op_name) {
        StringAttr name = builder_.getStringAttr(op_name);
        return NameLoc::get(name);
    }

    void insert_into_value_table(const Labs::Value *labs_value, Value mlir_value) {
        if (!value_table_.emplace(labs_value, mlir_value).second) {
            throw std::runtime_error("Value " + labs_value->name_ + " was defined twice");
        }
    }

    void gen_relu(const Labs::Op &op) {
        auto body = [](OpBuilder &b, Location loc, ValueRange args) {
            Value x = args[0];
            Value zero = arith::getZeroConstant(b, loc, x.getType());

            Value max_res = create_max(b, loc, x, zero);
            linalg::YieldOp::create(b, loc, max_res);
        };

        gen_elementwise(op, body);
    }

    void gen_add(const Labs::Op &op) {
        auto body = [](OpBuilder &b, Location loc, ValueRange args) {
            Value x = args[0], y = args[1];

            Value add_res = create_add(b, loc, x, y);
            linalg::YieldOp::create(b, loc, add_res);
        };

        gen_elementwise(op, body);
    }

    void gen_mul(const Labs::Op &op) {
        auto body = [](OpBuilder &b, Location loc, ValueRange args) {
            Value x = args[0], y = args[1];

            Value mul_res = create_mul(b, loc, x, y);
            linalg::YieldOp::create(b, loc, mul_res);
        };

        gen_elementwise(op, body);
    }

    void gen_matmul(const Labs::Op &op) {
        Value input1 = value_table_.at(op.inputs_[0]);
        Value input2 = value_table_.at(op.inputs_[1]);
        RankedTensorType outType = get_tensor_type(op.outputs_[0]);
        check_supported_type(outType.getElementType());

        std::size_t rank = op.outputs_[0]->shape_.size();
        Location location = make_loc(op.outputs_[0]->name_);
        Value init = make_zero_tensor(builder_, location, outType);

        SmallVector<AffineMap> maps = make_matmul_maps(op, rank);
        SmallVector<utils::IteratorType> iters = make_iters(rank, 1);

        auto body = [](OpBuilder &b, Location loc, ValueRange args) {
            Value x = args[0], y = args[1], out = args[2];

            Value mul_res = create_mul(b, loc, x, y);
            Value add_res = create_add(b, loc, out, mul_res);
            linalg::YieldOp::create(b, loc, add_res);
        };

        Value generic_res =
            create_generic(builder_, location, {input1, input2}, init, maps, iters, body);
        insert_into_value_table(op.outputs_[0], generic_res);
    }

    void gen_gemm(const Labs::Op &op) {
        const Labs::Gemm &gemm = static_cast<const Labs::Gemm &>(op);
        Value input1 = value_table_.at(gemm.inputs_[0]);
        Value input2 = value_table_.at(gemm.inputs_[1]);
        RankedTensorType outType = get_tensor_type(gemm.outputs_[0]);
        Type elem_type = outType.getElementType();
        if (!elem_type.isF32())
            throw std::runtime_error("Gemm is supported only for f32: " + gemm.outputs_[0]->name_);

        SmallVector<AffineMap> maps = make_gemm_maps(gemm);
        SmallVector<utils::IteratorType> iters = make_iters(2, 1);

        Location location = make_loc(gemm.outputs_[0]->name_);
        Value alpha =
            arith::ConstantOp::create(builder_, location, builder_.getF32FloatAttr(gemm.alpha_));
        Value init = make_gemm_init(gemm);

        auto body = [alpha](OpBuilder &b, Location loc, ValueRange args) {
            Value x = args[0], y = args[1], out = args[2];

            Value xy_mul = create_mul(b, loc, x, y);
            Value alpha_xy_mul = create_mul(b, loc, alpha, xy_mul);
            Value add_res = create_add(b, loc, out, alpha_xy_mul);
            linalg::YieldOp::create(b, loc, add_res);
        };

        Value generic_res =
            create_generic(builder_, location, {input1, input2}, init, maps, iters, body);
        insert_into_value_table(gemm.outputs_[0], generic_res);
    }

    void gen_conv(const Labs::Op &op) {
        const Labs::Conv &conv = static_cast<const Labs::Conv &>(op);
        Value X = make_padded_input(conv);
        Value K = value_table_.at(conv.inputs_[1]);
        Location location = make_loc(conv.outputs_[0]->name_);
        Type elem_type = getElementTypeOrSelf(X);
        check_supported_type(elem_type);

        int64_t G = conv.group_;
        Value X_new = expand_dim(X, G, 1, location);
        Value K_new = expand_dim(K, G, 0, location);

        auto [split_shape_Y, y_axis_groups] = split_dim(conv.outputs_[0]->shape_, G, 1);
        RankedTensorType split_Y_type = RankedTensorType::get(split_shape_Y, elem_type);
        Value init = make_zero_tensor(builder_, location, split_Y_type);

        std::size_t spatial_rank = conv.inputs_[0]->shape_.size() - 2;
        SmallVector<AffineMap> maps = make_conv_maps(conv, spatial_rank);
        SmallVector<utils::IteratorType> iters = make_iters(spatial_rank + 3, spatial_rank + 1);

        auto body = [](OpBuilder &b, Location loc, ValueRange args) {
            Value x = args[0], k = args[1], out = args[2];

            Value mul_res = create_mul(b, loc, x, k);
            Value add_res = create_add(b, loc, out, mul_res);
            linalg::YieldOp::create(b, loc, add_res);
        };

        RankedTensorType Y_type = get_tensor_type(conv.outputs_[0]);
        Value generic_res =
            create_generic(builder_, location, {X_new, K_new}, init, maps, iters, body);
        Value Y =
            tensor::CollapseShapeOp::create(builder_, location, Y_type, generic_res, y_axis_groups);

        if (conv.inputs_.size() == 3) {
            Value bias = value_table_.at(conv.inputs_[2]);
            Y = add_conv_bias(Y, bias, location);
        }

        insert_into_value_table(conv.outputs_[0], Y);
    }

    void gen_elementwise(const Labs::Op &op, const BodyBuilder &body) {
        RankedTensorType outType = get_tensor_type(op.outputs_[0]);
        Type elem_type = outType.getElementType();
        check_supported_type(elem_type);

        SmallVector<Value> inputs;
        SmallVector<AffineMap> maps;
        std::size_t rank = op.outputs_[0]->shape_.size();

        for (const auto *input_ptr : op.inputs_) {
            Value val_i = value_table_.at(input_ptr);

            maps.push_back(make_broadcast_map(input_ptr->shape_, op.outputs_[0]->shape_));
            inputs.push_back(val_i);
        }
        maps.push_back(builder_.getMultiDimIdentityMap(rank));

        Location location = make_loc(op.outputs_[0]->name_);
        Value init = tensor::EmptyOp::create(builder_, location, outType.getShape(), elem_type);
        SmallVector<utils::IteratorType> iters = make_iters(rank, 0);

        Value generic_res = create_generic(builder_, location, inputs, init, maps, iters, body);
        insert_into_value_table(op.outputs_[0], generic_res);
    }

    // Initial accumulator for Gemm: beta * C broadcast to the output shape,
    // or a zero tensor if C is absent.
    Value make_gemm_init(const Labs::Gemm &gemm) {
        RankedTensorType outType = get_tensor_type(gemm.outputs_[0]);
        Location location = make_loc(gemm.outputs_[0]->name_);

        if (gemm.inputs_.size() == 2)
            return make_zero_tensor(builder_, location, outType);

        const std::vector<int64_t> &out_shape = gemm.outputs_[0]->shape_;
        Value bias = value_table_.at(gemm.inputs_[2]);
        AffineMap bias_map = make_broadcast_map(gemm.inputs_[2]->shape_, out_shape);
        AffineMap output_map = builder_.getMultiDimIdentityMap(out_shape.size());
        SmallVector<AffineMap> maps = {bias_map, output_map};
        SmallVector<utils::IteratorType> iters = make_iters(out_shape.size(), 0);

        Value init = make_empty_tensor(builder_, location, outType);
        Value beta =
            arith::ConstantOp::create(builder_, location, builder_.getF32FloatAttr(gemm.beta_));

        auto body = [beta](OpBuilder &b, Location loc, ValueRange args) {
            Value bias = args[0];

            Value mul_res = create_mul(b, loc, beta, bias);
            linalg::YieldOp::create(b, loc, mul_res);
        };

        return create_generic(builder_, location, bias, init, maps, iters, body);
    }

    // prepares Value X: if necessary, performs padding
    Value make_padded_input(const Labs::Conv &conv) {
        const Labs::Value *labs_x = conv.inputs_[0];
        Value x = value_table_.at(labs_x);

        bool no_pad = std::ranges::all_of(conv.pads_, [](int64_t p) { return p == 0; });
        if (no_pad)
            return x;

        std::size_t spatial_rank = labs_x->shape_.size() - 2;
        std::vector<int64_t> padded_shape = labs_x->shape_;
        std::vector<int64_t> low_ints = {0, 0}, high_ints = {0, 0};

        for (std::size_t i = 0; i < spatial_rank; ++i) {
            int64_t begin = conv.pads_[i];
            int64_t end = conv.pads_[spatial_rank + i];

            padded_shape[2 + i] += (begin + end);
            low_ints.push_back(begin);
            high_ints.push_back(end);
        }

        Type elem_type = getElementTypeOrSelf(x);
        RankedTensorType padded_type = RankedTensorType::get(padded_shape, elem_type);

        auto low = getAsIndexOpFoldResult(builder_.getContext(), low_ints);
        auto high = getAsIndexOpFoldResult(builder_.getContext(), high_ints);

        Location location = make_loc(conv.outputs_[0]->name_);
        Value pad_value = arith::getZeroConstant(builder_, location, elem_type);

        return tensor::PadOp::create(builder_, location, padded_type, x, low, high, pad_value);
    }

    Value add_conv_bias(Value Y, Value bias, Location loc) {
        RankedTensorType Y_type = llvm::cast<RankedTensorType>(Y.getType());
        Value init =
            tensor::EmptyOp::create(builder_, loc, Y_type.getShape(), Y_type.getElementType());

        std::size_t rank = Y_type.getRank();
        SmallVector<utils::IteratorType> iters = make_iters(rank, 0);
        AffineExpr bias_expr = get_affine_expr(1);

        AffineMap bias_map = make_map(builder_, rank, bias_expr);
        AffineMap Y_map = builder_.getMultiDimIdentityMap(rank);
        AffineMap out_map = Y_map;
        SmallVector<AffineMap> maps = {Y_map, bias_map, out_map};

        auto body = [](OpBuilder &b, Location loc, ValueRange args) {
            Value conv_res = args[0], bias = args[1];

            Value add_res = create_add(b, loc, conv_res, bias);
            linalg::YieldOp::create(b, loc, add_res);
        };

        return create_generic(builder_, loc, {Y, bias}, init, maps, iters, body);
    }

    Value expand_dim(Value v, int64_t G, std::size_t pos, Location loc) {
        RankedTensorType type = llvm::cast<RankedTensorType>(v.getType());
        auto [split_shape, axis_groups] = split_dim(type.getShape(), G, pos);
        RankedTensorType split_type = RankedTensorType::get(split_shape, type.getElementType());

        return tensor::ExpandShapeOp::create(builder_, loc, split_type, v, axis_groups);
    }

    SmallVector<AffineMap> make_matmul_maps(const Labs::Op &op, std::size_t rank) {
        const std::vector<int64_t> &a_shape = op.inputs_[0]->shape_;
        const std::vector<int64_t> &b_shape = op.inputs_[1]->shape_;
        const std::vector<int64_t> &out_shape = op.outputs_[0]->shape_;
        bool a_is_vec = (a_shape.size() == 1);
        bool b_is_vec = (b_shape.size() == 1);

        std::size_t batch_rank = rank - (a_is_vec ? 0 : 1) - (b_is_vec ? 0 : 1);
        ArrayRef<int64_t> out_batch = ArrayRef(out_shape).take_front(batch_rank);

        AffineExpr dk = get_affine_expr(rank);
        AffineExpr dj = b_is_vec ? AffineExpr() : get_affine_expr(rank - 1);
        AffineExpr di = a_is_vec ? AffineExpr() : get_affine_expr(batch_rank);

        SmallVector<AffineExpr> input1_expr;
        if (a_is_vec) {
            input1_expr.push_back(dk);
        } else {
            input1_expr = make_broadcast_exprs(ArrayRef(a_shape).drop_back(2), out_batch);
            input1_expr.push_back(di);
            input1_expr.push_back(dk);
        }

        SmallVector<AffineExpr> input2_expr;
        if (b_is_vec) {
            input2_expr.push_back(dk);
        } else {
            input2_expr = make_broadcast_exprs(ArrayRef(b_shape).drop_back(2), out_batch);
            input2_expr.push_back(dk);
            input2_expr.push_back(dj);
        }

        SmallVector<AffineExpr> output_expr;
        for (std::size_t i = 0; i < rank; ++i) {
            output_expr.push_back(get_affine_expr(i));
        }

        std::size_t loops_amount = rank + 1;
        AffineMap input1_map = make_map(builder_, loops_amount, input1_expr);
        AffineMap input2_map = make_map(builder_, loops_amount, input2_expr);
        AffineMap output_map = make_map(builder_, loops_amount, output_expr);

        return {input1_map, input2_map, output_map};
    }

    SmallVector<AffineMap> make_gemm_maps(const Labs::Gemm &gemm) {
        // C[i, j] += alpha * A[i, k] * B[k, j] (A and B may be transposed)
        // i -> j -> k
        AffineExpr i = get_affine_expr(0);
        AffineExpr j = get_affine_expr(1);
        AffineExpr k = get_affine_expr(2);

        SmallVector<AffineExpr> input1_exprs, input2_exprs;
        if (gemm.need_transA_) {
            input1_exprs = {k, i};
        } else {
            input1_exprs = {i, k};
        }

        if (gemm.need_transB_) {
            input2_exprs = {j, k};
        } else {
            input2_exprs = {k, j};
        }

        AffineMap input1_map = make_map(builder_, 3, input1_exprs);
        AffineMap input2_map = make_map(builder_, 3, input2_exprs);
        AffineMap output_map = make_map(builder_, 3, {i, j});

        return {input1_map, input2_map, output_map};
    }

    SmallVector<AffineMap> make_conv_maps(const Labs::Conv &conv, std::size_t spatial_rank) {
        // counters of conv loop: n->g->f->oh1->oh2->...->ohn->c->kh1->kh2->...->khn
        const std::size_t n = 0, g = 1, f = 2;
        const std::size_t first_oh = 3;
        const std::size_t c = spatial_rank + 3;
        const std::size_t first_kh = spatial_rank + 4;

        SmallVector<AffineExpr> X_exprs{get_affine_expr(n), get_affine_expr(g), get_affine_expr(c)};
        for (std::size_t i = 0; i < spatial_rank; ++i) {
            AffineExpr current = get_affine_expr(i + first_oh) * conv.strides_[i] +
                                 get_affine_expr(i + first_kh) * conv.dilations_[i];

            X_exprs.push_back(current);
        }

        SmallVector<AffineExpr> K_exprs{get_affine_expr(g), get_affine_expr(f), get_affine_expr(c)};
        for (std::size_t i = 0; i < spatial_rank; ++i)
            K_exprs.push_back(get_affine_expr(first_kh + i));

        SmallVector<AffineExpr> output_exprs{get_affine_expr(n), get_affine_expr(g),
                                             get_affine_expr(f)};
        for (std::size_t i = 0; i < spatial_rank; ++i)
            output_exprs.push_back(get_affine_expr(first_oh + i));

        AffineMap X_map, K_map, output_map;
        std::size_t loops_amount = 2 * spatial_rank + 4;
        X_map = make_map(builder_, loops_amount, X_exprs);
        K_map = make_map(builder_, loops_amount, K_exprs);
        output_map = make_map(builder_, loops_amount, output_exprs);

        return {X_map, K_map, output_map};
    }

    // Full broadcast map over output.size() loop dims.
    AffineMap make_broadcast_map(ArrayRef<int64_t> input, ArrayRef<int64_t> output) {
        if (input == output)
            return builder_.getMultiDimIdentityMap(input.size());

        SmallVector<AffineExpr> exprs = make_broadcast_exprs(input, output);
        return make_map(builder_, output.size(), exprs);
    }

    // Broadcast index expressions for `input` aligned to `output` (NumPy-style):
    // missing leading dims are skipped, size-1 dims map to 0.
    // For MatMul pass only batch dims!
    SmallVector<AffineExpr> make_broadcast_exprs(ArrayRef<int64_t> input,
                                                 ArrayRef<int64_t> output) {
        SmallVector<AffineExpr> input_affine_expr;
        std::size_t size_diff = output.size() - input.size();

        for (std::size_t i = 0; i < input.size(); ++i) {
            bool dim_equal_unit = (input[i] == 1);
            AffineExpr cur_expr = (dim_equal_unit) ? builder_.getAffineConstantExpr(0)
                                                   : get_affine_expr(size_diff + i);
            input_affine_expr.push_back(cur_expr);
        }
        return input_affine_expr;
    }

    AffineExpr get_affine_expr(std::size_t num) { return builder_.getAffineDimExpr(num); }

}; // class MLIRGenImpl

SmallVector<int64_t> to_mlir_shape(const std::vector<int64_t> &val_shape) {
    SmallVector<int64_t> shape(val_shape.begin(), val_shape.end());
    int64_t absent_dim = -1;

    std::replace(shape.begin(), shape.end(), absent_dim, ShapedType::kDynamic);
    return shape;
}

SplitResult split_dim(ArrayRef<int64_t> shape, int64_t num_groups, std::size_t pos) {
    std::vector<int64_t> split_shape = shape;
    split_shape[pos] /= num_groups;
    split_shape.insert(split_shape.begin() + pos, num_groups);

    SmallVector<ReassociationIndices> axis_groups;
    for (int64_t i = 0; i < shape.size(); ++i) {
        if (i < pos) {
            axis_groups.push_back({i});
        } else if (i == pos) {
            axis_groups.push_back({i, i + 1});
        } else {
            axis_groups.push_back({i + 1});
        }
    }
    return {std::move(split_shape), std::move(axis_groups)};
}

bool is_dynamic(const Labs::Value *val) {
    auto dynamic_it = std::find(val->shape_.begin(), val->shape_.end(), -1);
    return (dynamic_it != val->shape_.end());
}

void check_static_shapes(const Labs::Op *op) {
    auto check = [](const Labs::Value *val) {
        if (is_dynamic(val))
            throw std::runtime_error("dynamic shapes are not supported yet: " + val->name_);
    };

    std::for_each(op->inputs_.begin(), op->inputs_.end(), check);
    std::for_each(op->outputs_.begin(), op->outputs_.end(), check);
}

void check_supported_type(Type elem_type) {
    if (!elem_type.isF32() && !elem_type.isInteger(64)) {
        throw std::runtime_error(
            llvm::formatv("Unsupported type of elements: {0}", elem_type).str());
    }
}

Value create_generic(OpBuilder &b, Location loc, ValueRange inputs, Value init,
                     ArrayRef<AffineMap> maps, ArrayRef<utils::IteratorType> iters,
                     const BodyBuilder &body) {
    linalg::GenericOp generic =
        linalg::GenericOp::create(b, loc, init.getType(), inputs, init, maps, iters, body);
    Value generic_res = generic.getResult(0);

    return generic_res;
}

Value create_add(OpBuilder &b, Location loc, Value x, Value y) {
    if (x.getType().isF32()) {
        return arith::AddFOp::create(b, loc, x, y);
    }

    return arith::AddIOp::create(b, loc, x, y);
}

Value create_mul(OpBuilder &b, Location loc, Value x, Value y) {
    if (x.getType().isF32()) {
        return arith::MulFOp::create(b, loc, x, y);
    }

    return arith::MulIOp::create(b, loc, x, y);
}

Value create_max(OpBuilder &b, Location loc, Value x, Value y) {
    if (x.getType().isF32()) {
        return arith::MaximumFOp::create(b, loc, x, y);
    }

    return arith::MaxSIOp::create(b, loc, x, y);
}

SmallVector<utils::IteratorType> make_iters(std::size_t n_par, std::size_t n_red) {
    SmallVector<utils::IteratorType> iters;
    iters.append(n_par, utils::IteratorType::parallel);
    iters.append(n_red, utils::IteratorType::reduction);

    return iters;
}

AffineMap make_map(OpBuilder &b, std::size_t loops_amount, ArrayRef<AffineExpr> exprs) {
    return AffineMap::get(loops_amount, 0, exprs, b.getContext());
}

Value make_empty_tensor(OpBuilder &b, Location loc, RankedTensorType type) {
    return tensor::EmptyOp::create(b, loc, type.getShape(), type.getElementType());
}

Value make_zero_tensor(OpBuilder &b, Location loc, RankedTensorType type) {
    Value zero = arith::getZeroConstant(b, loc, type.getElementType());
    Value empty = make_empty_tensor(b, loc, type);
    Value zeros = linalg::FillOp::create(b, loc, type, zero, empty).getResult(0);

    return zeros;
}

} // namespace

namespace Labs {
OwningOpRef<ModuleOp> mlirGen(MLIRContext &context, const ONNX_Graph &graph) {
    return MLIRGenImpl(context).mlirGen(graph);
}
} // namespace Labs