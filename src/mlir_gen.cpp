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
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LogicalResult.h"
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
using BodyBuilder = std::function<void(OpBuilder &, Location, ValueRange)>;

SmallVector<int64_t> convert_shape_to_mlir(const std::vector<int64_t> &val_shape);
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
Value make_zero_tensor(OpBuilder &b, Location loc, RankedTensorType out_type);

class MLIRGenImpl {
    using SplitResult = std::pair<std::vector<int64_t>, SmallVector<ReassociationIndices>>;

  public:
    MLIRGenImpl(MLIRContext &context) : builder(&context) {}

    OwningOpRef<ModuleOp> mlirGen(const Labs::ONNX_Graph &my_graph) {
        OwningOpRef owner = ModuleOp::create(builder.getUnknownLoc());
        theModule = owner.get();

        func::FuncOp main = create_main(my_graph.inputs, my_graph.outputs);
        builder.setInsertionPointToStart(&main.front());
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
                gen_conv(static_cast<Labs::Conv &>(*op));
            } else {
                throw std::runtime_error("unsupported gen operation now: " + op_type);
            }
        }

        SmallVector<Value> results;
        for (const auto *out_val : my_graph.outputs) {
            results.push_back(valueTable.at(out_val));
        }
        func::ReturnOp::create(builder, builder.getUnknownLoc(), results);

        return owner;
    }

    func::FuncOp create_main(const std::vector<Labs::Value *> &inputs,
                             const std::vector<Labs::Value *> &outputs) {
        Location location = convert_str_to_loc("onnx_graph");

        SmallVector<Type> argsTypes = get_types(inputs);
        SmallVector<Type> resTypes = get_types(outputs);

        FunctionType funcType = builder.getFunctionType(argsTypes, resTypes);
        builder.setInsertionPointToStart(theModule.getBody());
        func::FuncOp main = func::FuncOp::create(builder, location, "main", funcType);

        Block *entry = main.addEntryBlock();

        assert(inputs.size() == entry->getNumArguments());
        for (std::size_t i = 0; i < inputs.size(); ++i)
            valueTable_insertion(inputs[i], entry->getArgument(i));

        return main;
    }

  private:
    ModuleOp theModule;
    OpBuilder builder;
    std::unordered_map<const Labs::Value *, Value> valueTable;

    RankedTensorType get_type(const Labs::Value *value) {
        Type mlir_type;
        switch (value->type_) {
        case Labs::DataType::Float32:
            mlir_type = builder.getF32Type();
            break;
        case Labs::DataType::Int64:
            mlir_type = builder.getI64Type();
            break;
        default:
            throw std::runtime_error("Unsupported type");
        }

        SmallVector<int64_t> mlir_shape = convert_shape_to_mlir(value->shape_);
        return RankedTensorType::get(mlir_shape, mlir_type);
    }

    SmallVector<Type> get_types(const std::vector<Labs::Value *> &vec) {
        SmallVector<Type> vecTypes;
        vecTypes.reserve(vec.size());

        for (const auto *value : vec) {
            vecTypes.push_back(get_type(value));
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
        RankedTensorType type = get_type(value_ptr);
        const char *raw_array = reinterpret_cast<const char *>(value_ptr->raw_data_.data());
        ArrayRef<char> bytes(raw_array, value_ptr->raw_data_.size());
        DenseElementsAttr attr = DenseElementsAttr::getFromRawBuffer(type, bytes);

        Location location = convert_str_to_loc("main");
        Value cnst = arith::ConstantOp::create(builder, location, attr);
        valueTable_insertion(value_ptr, cnst);
    }

    Location convert_str_to_loc(const std::string &op_name) {
        StringAttr name = builder.getStringAttr(op_name);
        return NameLoc::get(name);
    }

    void valueTable_insertion(const Labs::Value *labs_value, Value mlir_value) {
        if (!valueTable.emplace(labs_value, mlir_value).second) {
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
        Value input1 = valueTable.at(op.inputs_[0]);
        Value input2 = valueTable.at(op.inputs_[1]);
        RankedTensorType outType = get_type(op.outputs_[0]);
        Type elem_type = outType.getElementType();
        check_supported_type(elem_type);

        std::size_t rank = op.outputs_[0]->shape_.size();
        Location location = convert_str_to_loc(op.outputs_[0]->name_);
        Value init = make_zero_tensor(builder, location, outType);

        SmallVector<utils::IteratorType> iters = make_iters(rank, 1);

        const std::vector<int64_t> &shapeA = op.inputs_[0]->shape_;
        const std::vector<int64_t> &shapeB = op.inputs_[1]->shape_;
        const std::vector<int64_t> &shapeC = op.outputs_[0]->shape_;
        bool a_is_vec = (shapeA.size() == 1);
        bool b_is_vec = (shapeB.size() == 1);
        std::size_t output_offset = ((a_is_vec) ? 0 : 1) + ((b_is_vec) ? 0 : 1);

        AffineExpr dk = get_affine_expr(rank);
        AffineExpr di, dj;
        if (!b_is_vec)
            dj = get_affine_expr(rank - 1);
        if (!a_is_vec)
            di = get_affine_expr(b_is_vec ? rank - 1 : rank - 2);

        SmallVector<AffineExpr> input1_expr;
        if (a_is_vec) {
            input1_expr.push_back(dk);
        } else {
            input1_expr = get_broadcast_exprs(ArrayRef(shapeA).drop_back(2),
                                              ArrayRef(shapeC).drop_back(output_offset));
            input1_expr.push_back(di);
            input1_expr.push_back(dk);
        }

        SmallVector<AffineExpr> input2_expr;
        if (b_is_vec) {
            input2_expr.push_back(dk);
        } else {
            input2_expr = get_broadcast_exprs(ArrayRef(shapeB).drop_back(2),
                                              ArrayRef(shapeC).drop_back(output_offset));
            input2_expr.push_back(dk);
            input2_expr.push_back(dj);
        }

        SmallVector<AffineExpr> output_expr;
        for (std::size_t i = 0; i < rank; ++i) {
            output_expr.push_back(get_affine_expr(i));
        }

        AffineMap input1_map = make_map(builder, rank + 1, input1_expr);
        AffineMap input2_map = make_map(builder, rank + 1, input2_expr);
        AffineMap output_map = make_map(builder, rank + 1, output_expr);
        SmallVector<AffineMap> maps = {input1_map, input2_map, output_map};

        auto body = [](OpBuilder &b, Location loc, ValueRange args) {
            Value x = args[0], y = args[1], out = args[2];

            Value mul_res = create_mul(b, loc, x, y);
            Value add_res = create_add(b, loc, out, mul_res);
            linalg::YieldOp::create(b, loc, add_res);
        };

        Value generic_res =
            create_generic(builder, location, {input1, input2}, init, maps, iters, body);
        valueTable_insertion(op.outputs_[0], generic_res);
    }

    void gen_gemm(const Labs::Op &op) {
        Value input1 = valueTable.at(op.inputs_[0]);
        Value input2 = valueTable.at(op.inputs_[1]);
        RankedTensorType outType = get_type(op.outputs_[0]);
        Type elem_type = outType.getElementType();
        if (!elem_type.isF32())
            throw std::runtime_error("Gemm is supported only for f32: " + op.outputs_[0]->name_);

        const Labs::Gemm &gemm = static_cast<const Labs::Gemm &>(op);
        Value init = gen_gemm_init(op);
        SmallVector<utils::IteratorType> iters = make_iters(2, 1);

        AffineExpr d0 = get_affine_expr(0);
        AffineExpr d1 = get_affine_expr(1);
        AffineExpr d2 = get_affine_expr(2);
        SmallVector<AffineExpr> input1_exprs, input2_exprs, output_exprs;
        output_exprs = {d0, d1};
        if (gemm.need_transA_) {
            input1_exprs = {d2, d0};
        } else {
            input1_exprs = {d0, d2};
        }
        if (gemm.need_transB_) {
            input2_exprs = {d1, d2};
        } else {
            input2_exprs = {d2, d1};
        }

        AffineMap input1_map = make_map(builder, 3, input1_exprs);
        AffineMap input2_map = make_map(builder, 3, input2_exprs);
        AffineMap output_map = make_map(builder, 3, output_exprs);
        SmallVector<AffineMap> maps = {input1_map, input2_map, output_map};

        Location location = convert_str_to_loc(gemm.outputs_[0]->name_);
        Value alpha =
            arith::ConstantOp::create(builder, location, builder.getF32FloatAttr(gemm.alpha_));

        auto body = [alpha](OpBuilder &b, Location loc, ValueRange args) {
            Value x = args[0], y = args[1], out = args[2];

            Value xy_mul = create_mul(b, loc, x, y);
            Value alpha_xy_mul = create_mul(b, loc, alpha, xy_mul);
            Value add_res = create_add(b, loc, out, alpha_xy_mul);
            linalg::YieldOp::create(b, loc, add_res);
        };

        Value generic_res =
            create_generic(builder, location, {input1, input2}, init, maps, iters, body);
        valueTable_insertion(gemm.outputs_[0], generic_res);
    }

    void gen_conv(const Labs::Conv &conv) {
        Value X = gen_conv_init(conv);
        Value K = valueTable.at(conv.inputs_[1]);
        int64_t G = conv.group_;
        Location location = convert_str_to_loc(conv.outputs_[0]->name_);
        Type elem_type = getElementTypeOrSelf(X);
        check_supported_type(elem_type);

        RankedTensorType X_padded_type = llvm::cast<RankedTensorType>(X.getType());
        auto [split_shape_X, split_groups_X] = split_dim(X_padded_type.getShape(), G, 1);
        RankedTensorType X_split_type = RankedTensorType::get(split_shape_X, elem_type);
        Value X_new =
            tensor::ExpandShapeOp::create(builder, location, X_split_type, X, split_groups_X);

        auto [split_shape_K, split_groups_K] = split_dim(conv.inputs_[1]->shape_, G, 0);
        RankedTensorType K_split_type = RankedTensorType::get(split_shape_K, elem_type);
        Value K_new =
            tensor::ExpandShapeOp::create(builder, location, K_split_type, K, split_groups_K);

        auto [split_shape_Y, split_groups_y] = split_dim(conv.outputs_[0]->shape_, G, 1);
        RankedTensorType split_Y_type = RankedTensorType::get(split_shape_Y, elem_type);
        Value init = make_zero_tensor(builder, location, split_Y_type);

        std::size_t dims_amount = conv.inputs_[0]->shape_.size() - 2;
        SmallVector<utils::IteratorType> iters = make_iters(dims_amount + 3, dims_amount + 1);

        SmallVector<AffineExpr> output_exprs;
        for (std::size_t i = 0; i < 3 + dims_amount; ++i) {
            output_exprs.push_back(get_affine_expr(i));
        }

        SmallVector<AffineExpr> K_exprs;
        K_exprs.push_back(get_affine_expr(1));
        K_exprs.push_back(get_affine_expr(2));
        for (std::size_t i = dims_amount + 3; i < 2 * dims_amount + 4; ++i) {
            K_exprs.push_back(get_affine_expr(i));
        }

        SmallVector<AffineExpr> X_exprs;
        X_exprs.push_back(get_affine_expr(0));
        X_exprs.push_back(get_affine_expr(1));
        X_exprs.push_back(get_affine_expr(dims_amount + 3));
        for (std::size_t i = 0; i < dims_amount; ++i) {
            AffineExpr cur = get_affine_expr(i + 3) * conv.strides_[i] +
                             get_affine_expr(i + dims_amount + 4) * conv.dilations_[i];
            X_exprs.push_back(cur);
        }

        AffineMap out_map, K_map, X_map;
        out_map = make_map(builder, 2 * dims_amount + 4, output_exprs);
        K_map = make_map(builder, 2 * dims_amount + 4, K_exprs);
        X_map = make_map(builder, 2 * dims_amount + 4, X_exprs);
        SmallVector<AffineMap> maps = {X_map, K_map, out_map};

        auto body = [](OpBuilder &b, Location loc, ValueRange args) {
            Value x = args[0], k = args[1], out = args[2];

            Value mul_res = create_mul(b, loc, x, k);
            Value add_res = create_add(b, loc, out, mul_res);
            linalg::YieldOp::create(b, loc, add_res);
        };

        RankedTensorType Y_type = get_type(conv.outputs_[0]);
        Value generic_res =
            create_generic(builder, location, {X_new, K_new}, init, maps, iters, body);
        Value Y =
            tensor::CollapseShapeOp::create(builder, location, Y_type, generic_res, split_groups_y);
        Value table_insertion = Y;

        //------------------------------------------------------------------------------------------
        if (conv.inputs_.size() == 3) {
            Value bias = valueTable.at(conv.inputs_[2]);
            Value bias_init =
                tensor::EmptyOp::create(builder, location, Y_type.getShape(), elem_type);
            SmallVector<utils::IteratorType> bias_iters = make_iters(dims_amount + 2, 0);
            AffineExpr bias_exprs = get_affine_expr(1);

            AffineMap bias_map = make_map(builder, 2 + dims_amount, bias_exprs);
            AffineMap Y_map = builder.getMultiDimIdentityMap(2 + dims_amount);
            AffineMap bias_out_map = Y_map;
            SmallVector<AffineMap> bias_maps = {Y_map, bias_map, bias_out_map};

            auto bias_body = [](OpBuilder &b, Location loc, ValueRange args) {
                Value conv_res = args[0], bias = args[1];

                Value add_res = create_add(b, loc, conv_res, bias);
                linalg::YieldOp::create(b, loc, add_res);
            };

            Value bias_generic_res = create_generic(builder, location, {Y, bias}, bias_init,
                                                    bias_maps, bias_iters, bias_body);
            table_insertion = bias_generic_res;
        }

        valueTable_insertion(conv.outputs_[0], table_insertion);
    }

    void gen_elementwise(const Labs::Op &op, const BodyBuilder &body) {
        RankedTensorType outType = get_type(op.outputs_[0]);
        Type elem_type = outType.getElementType();
        check_supported_type(elem_type);

        SmallVector<Value> inputs;
        SmallVector<AffineMap> maps;
        std::size_t rank = op.outputs_[0]->shape_.size();

        for (const auto *input_ptr : op.inputs_) {
            Value val_i = valueTable.at(input_ptr);

            maps.push_back(get_broadcast_map(input_ptr->shape_, op.outputs_[0]->shape_));
            inputs.push_back(val_i);
        }
        maps.push_back(builder.getMultiDimIdentityMap(rank));

        Location location = convert_str_to_loc(op.outputs_[0]->name_);
        Value init = tensor::EmptyOp::create(builder, location, outType.getShape(), elem_type);
        SmallVector<utils::IteratorType> iters = make_iters(rank, 0);

        Value generic_res = create_generic(builder, location, inputs, init, maps, iters, body);
        valueTable_insertion(op.outputs_[0], generic_res);
    }

    // Prepares Value equal to the bias of the desired size
    Value gen_gemm_init(const Labs::Op &op) {
        RankedTensorType outType = get_type(op.outputs_[0]);
        Type elem_type = outType.getElementType();
        Location location = convert_str_to_loc(op.outputs_[0]->name_);

        if (op.inputs_.size() == 2) {
            Value zeros = make_zero_tensor(builder, location, outType);

            return zeros;
        }

        Value init = tensor::EmptyOp::create(builder, location, outType.getShape(), elem_type);
        Value bias = valueTable.at(op.inputs_[2]);
        AffineMap bias_map = get_broadcast_map(op.inputs_[2]->shape_, op.outputs_[0]->shape_);
        AffineMap output_map = builder.getMultiDimIdentityMap(op.outputs_[0]->shape_.size());
        SmallVector<AffineMap> maps = {bias_map, output_map};
        SmallVector<utils::IteratorType> iters = make_iters(op.outputs_[0]->shape_.size(), 0);

        const Labs::Gemm &gemm = static_cast<const Labs::Gemm &>(op);
        Value beta =
            arith::ConstantOp::create(builder, location, builder.getF32FloatAttr(gemm.beta_));
        auto body = [beta](OpBuilder &b, Location loc, ValueRange args) {
            Value bias = args[0];

            Value mul_res = create_mul(b, loc, beta, bias);
            linalg::YieldOp::create(b, loc, mul_res);
        };

        Value generic_res = create_generic(builder, location, bias, init, maps, iters, body);
        return generic_res;
    }

    // prepares Value X: if necessary, performs padding
    Value gen_conv_init(const Labs::Conv &conv) {
        bool no_pad = std::ranges::all_of(conv.pads_, [](int64_t p) { return p == 0; });

        if (no_pad) {
            return valueTable.at(conv.inputs_[0]);
        }

        std::size_t spatial_rank = conv.inputs_[0]->shape_.size() - 2;
        std::vector<int64_t> padded_shape = conv.inputs_[0]->shape_;
        std::vector<int64_t> low_ints = {0, 0}, high_ints = {0, 0};
        for (std::size_t i = 0; i < spatial_rank; ++i) {
            padded_shape[2 + i] += conv.pads_[i];
            padded_shape[2 + i] += conv.pads_[i + spatial_rank];
            low_ints.push_back(conv.pads_[i]);
            high_ints.push_back(conv.pads_[i + spatial_rank]);
        }

        RankedTensorType temp_type = get_type(conv.inputs_[0]);
        Type elem_type = temp_type.getElementType();
        RankedTensorType padded_type = RankedTensorType::get(padded_shape, elem_type);
        Location location = convert_str_to_loc(conv.outputs_[0]->name_);
        Value x = valueTable.at(conv.inputs_[0]);
        auto low = getAsIndexOpFoldResult(builder.getContext(), low_ints);
        auto high = getAsIndexOpFoldResult(builder.getContext(), high_ints);
        Value pad_value;
        pad_value = arith::getZeroConstant(builder, location, elem_type);

        Value result =
            tensor::PadOp::create(builder, location, padded_type, x, low, high, pad_value);
        return result;
    }

    // Full broadcast map over output.size() loop dims.
    AffineMap get_broadcast_map(ArrayRef<int64_t> input, ArrayRef<int64_t> output) {
        if (input == output)
            return builder.getMultiDimIdentityMap(input.size());

        SmallVector<AffineExpr> exprs = get_broadcast_exprs(input, output);
        return make_map(builder, output.size(), exprs);
    }

    // Broadcast index expressions for `input` aligned to `output` (NumPy-style):
    // missing leading dims are skipped, size-1 dims map to 0.
    // For MatMul pass only batch dims!
    SmallVector<AffineExpr> get_broadcast_exprs(ArrayRef<int64_t> input, ArrayRef<int64_t> output) {
        SmallVector<AffineExpr> input_affine_expr;
        std::size_t size_diff = output.size() - input.size();

        for (std::size_t i = 0; i < input.size(); ++i) {
            bool dim_equal_unit = (input[i] == 1);
            AffineExpr cur_expr = (dim_equal_unit) ? builder.getAffineConstantExpr(0)
                                                   : get_affine_expr(size_diff + i);
            input_affine_expr.push_back(cur_expr);
        }
        return input_affine_expr;
    }

    SplitResult split_dim(const std::vector<int64_t> &shape, const std::size_t G,
                          const std::size_t pos) {
        std::vector<int64_t> split_shape = shape;
        split_shape[pos] /= G;
        split_shape.insert(split_shape.begin() + pos, G);

        SmallVector<ReassociationIndices> groups;
        for (int64_t i = 0; i < shape.size(); ++i) {
            if (i < pos) {
                groups.push_back({i});
            } else if (i == pos) {
                groups.push_back({i, i + 1});
            } else {
                groups.push_back({i + 1});
            }
        }
        return {std::move(split_shape), std::move(groups)};
    }

    AffineExpr get_affine_expr(std::size_t num) { return builder.getAffineDimExpr(num); }

}; // class MLIRGenImpl

SmallVector<int64_t> convert_shape_to_mlir(const std::vector<int64_t> &val_shape) {
    SmallVector<int64_t> shape(val_shape.begin(), val_shape.end());
    int64_t absent_dim = -1;

    std::replace(shape.begin(), shape.end(), absent_dim, ShapedType::kDynamic);
    return shape;
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

    return arith::MulFOp::create(b, loc, x, y);
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

Value make_zero_tensor(OpBuilder &b, Location loc, RankedTensorType out_type) {
    Value zero = arith::getZeroConstant(b, loc, out_type.getElementType());
    Value empty = tensor::EmptyOp::create(b, loc, out_type.getShape(), out_type.getElementType());
    Value zeros = linalg::FillOp::create(b, loc, out_type, zero, empty).getResult(0);

    return zeros;
}

} // namespace

namespace Labs {
OwningOpRef<ModuleOp> mlirGen(MLIRContext &context, const ONNX_Graph &graph) {
    return MLIRGenImpl(context).mlirGen(graph);
}
} // namespace Labs