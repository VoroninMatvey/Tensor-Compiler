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

namespace {
mlir::SmallVector<int64_t> convert_shape_to_mlir(const std::vector<int64_t> &val_shape);
void check_supported_type(mlir::Type elem_type);
void check_static_shapes(const Labs::Op *op);
bool is_dynamic(const Labs::Value *val);

class MLIRGenImpl {
    using BodyBuilder = std::function<void(mlir::OpBuilder &, mlir::Location, mlir::ValueRange)>;
    using SplitResult =
        std::pair<std::vector<int64_t>, mlir::SmallVector<mlir::ReassociationIndices>>;

  public:
    MLIRGenImpl(mlir::MLIRContext &context) : builder(&context) {}

    mlir::OwningOpRef<mlir::ModuleOp> mlirGen(const Labs::ONNX_Graph &my_graph) {
        mlir::OwningOpRef owner = mlir::ModuleOp::create(builder.getUnknownLoc());
        theModule = owner.get();

        mlir::func::FuncOp main = create_main(my_graph.inputs, my_graph.outputs);
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

        mlir::SmallVector<mlir::Value> results;
        for (const auto *out_val : my_graph.outputs) {
            results.push_back(valueTable.at(out_val));
        }
        mlir::func::ReturnOp::create(builder, builder.getUnknownLoc(), results);

        return owner;
    }

    mlir::func::FuncOp create_main(const std::vector<Labs::Value *> &inputs,
                                   const std::vector<Labs::Value *> &outputs) {
        mlir::Location location = convert_str_to_loc("onnx_graph");

        mlir::SmallVector<mlir::Type> argsTypes = get_types(inputs);
        mlir::SmallVector<mlir::Type> resTypes = get_types(outputs);

        mlir::FunctionType funcType = builder.getFunctionType(argsTypes, resTypes);
        builder.setInsertionPointToStart(theModule.getBody());
        mlir::func::FuncOp main = mlir::func::FuncOp::create(builder, location, "main", funcType);

        mlir::Block *entry = main.addEntryBlock();

        assert(inputs.size() == entry->getNumArguments());
        for (std::size_t i = 0; i < inputs.size(); ++i)
            valueTable_insertion(inputs[i], entry->getArgument(i));

        return main;
    }

  private:
    mlir::ModuleOp theModule;
    mlir::OpBuilder builder;
    std::unordered_map<const Labs::Value *, mlir::Value> valueTable;

    mlir::RankedTensorType get_type(const Labs::Value *value) {
        mlir::Type mlir_type;
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

        mlir::SmallVector<int64_t> mlir_shape = convert_shape_to_mlir(value->shape_);
        return mlir::RankedTensorType::get(mlir_shape, mlir_type);
    }

    mlir::SmallVector<mlir::Type> get_types(const std::vector<Labs::Value *> &vec) {
        mlir::SmallVector<mlir::Type> vecTypes;
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
        mlir::RankedTensorType type = get_type(value_ptr);
        const char *raw_array = reinterpret_cast<const char *>(value_ptr->raw_data_.data());
        mlir::ArrayRef<char> bytes(raw_array, value_ptr->raw_data_.size());
        mlir::DenseElementsAttr attr = mlir::DenseElementsAttr::getFromRawBuffer(type, bytes);

        mlir::Location location = convert_str_to_loc("main");
        mlir::Value cnst = mlir::arith::ConstantOp::create(builder, location, attr);
        valueTable_insertion(value_ptr, cnst);
    }

    mlir::Location convert_str_to_loc(const std::string &op_name) {
        mlir::StringAttr name = builder.getStringAttr(op_name);
        return mlir::NameLoc::get(name);
    }

    void valueTable_insertion(const Labs::Value *labs_value, mlir::Value mlir_value) {
        if (!valueTable.emplace(labs_value, mlir_value).second) {
            throw std::runtime_error("Value " + labs_value->name_ + " was defined twice");
        }
    }

    void gen_relu(const Labs::Op &op) {
        mlir::Type elem_type = get_type(op.outputs_[0]).getElementType();

        auto body = [elem_type](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange args) {
            mlir::Value zero = mlir::arith::getZeroConstant(b, loc, elem_type);

            mlir::Value res;
            if (elem_type.isF32()) {
                res = mlir::arith::MaximumFOp::create(b, loc, args[0], zero);
            } else {
                res = mlir::arith::MaxSIOp::create(b, loc, args[0], zero);
            }

            mlir::linalg::YieldOp::create(b, loc, res);
        };

        gen_elementwise(op, body);
    }

    void gen_add(const Labs::Op &op) {
        mlir::Type elem_type = get_type(op.outputs_[0]).getElementType();

        auto body = [elem_type](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange args) {
            mlir::Value res;
            if (elem_type.isF32()) {
                res = mlir::arith::AddFOp::create(b, loc, args[0], args[1]);
            } else {
                res = mlir::arith::AddIOp::create(b, loc, args[0], args[1]);
            }

            mlir::linalg::YieldOp::create(b, loc, res);
        };

        gen_elementwise(op, body);
    }

    void gen_mul(const Labs::Op &op) {
        mlir::Type elem_type = get_type(op.outputs_[0]).getElementType();

        auto body = [elem_type](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange args) {
            mlir::Value res;
            if (elem_type.isF32()) {
                res = mlir::arith::MulFOp::create(b, loc, args[0], args[1]);
            } else {
                res = mlir::arith::MulIOp::create(b, loc, args[0], args[1]);
            }

            mlir::linalg::YieldOp::create(b, loc, res);
        };

        gen_elementwise(op, body);
    }

    void gen_matmul(const Labs::Op &op) {
        mlir::Value input1 = valueTable.at(op.inputs_[0]);
        mlir::Value input2 = valueTable.at(op.inputs_[1]);
        mlir::RankedTensorType outType = get_type(op.outputs_[0]);
        mlir::Type elem_type = outType.getElementType();
        check_supported_type(elem_type);

        std::size_t rank = op.outputs_[0]->shape_.size();
        mlir::Location location = convert_str_to_loc(op.outputs_[0]->name_);
        mlir::Value zero = mlir::arith::getZeroConstant(builder, location, elem_type);
        mlir::Value empty =
            mlir::tensor::EmptyOp::create(builder, location, outType.getShape(), elem_type);
        mlir::Value init =
            mlir::linalg::FillOp::create(builder, location, outType, zero, empty).getResult(0);

        mlir::SmallVector<mlir::utils::IteratorType> iters(rank,
                                                           mlir::utils::IteratorType::parallel);
        iters.push_back(mlir::utils::IteratorType::reduction);

        const std::vector<int64_t> &shapeA = op.inputs_[0]->shape_;
        const std::vector<int64_t> &shapeB = op.inputs_[1]->shape_;
        const std::vector<int64_t> &shapeC = op.outputs_[0]->shape_;
        bool a_is_vec = (shapeA.size() == 1);
        bool b_is_vec = (shapeB.size() == 1);
        std::size_t output_offset = ((a_is_vec) ? 0 : 1) + ((b_is_vec) ? 0 : 1);

        auto expr_dim = [this](std::size_t rank) { return builder.getAffineDimExpr(rank); };
        mlir::AffineExpr dk = builder.getAffineDimExpr(rank);
        mlir::AffineExpr di, dj;
        if (!b_is_vec)
            dj = expr_dim(rank - 1);
        if (!a_is_vec)
            di = expr_dim(b_is_vec ? rank - 1 : rank - 2);

        mlir::SmallVector<mlir::AffineExpr> input1_expr;
        if (a_is_vec) {
            input1_expr.push_back(dk);
        } else {
            input1_expr = get_broadcast_exprs(mlir::ArrayRef(shapeA).drop_back(2),
                                              mlir::ArrayRef(shapeC).drop_back(output_offset));
            input1_expr.push_back(di);
            input1_expr.push_back(dk);
        }

        mlir::SmallVector<mlir::AffineExpr> input2_expr;
        if (b_is_vec) {
            input2_expr.push_back(dk);
        } else {
            input2_expr = get_broadcast_exprs(mlir::ArrayRef(shapeB).drop_back(2),
                                              mlir::ArrayRef(shapeC).drop_back(output_offset));
            input2_expr.push_back(dk);
            input2_expr.push_back(dj);
        }

        mlir::SmallVector<mlir::AffineExpr> output_expr;
        for (std::size_t i = 0; i < rank; ++i) {
            output_expr.push_back(builder.getAffineDimExpr(i));
        }

        mlir::AffineMap input1_map =
            mlir::AffineMap::get(rank + 1, 0, input1_expr, builder.getContext());
        mlir::AffineMap input2_map =
            mlir::AffineMap::get(rank + 1, 0, input2_expr, builder.getContext());
        mlir::AffineMap output_map =
            mlir::AffineMap::get(rank + 1, 0, output_expr, builder.getContext());
        mlir::SmallVector<mlir::AffineMap> maps = {input1_map, input2_map, output_map};

        auto body = [elem_type](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange args) {
            mlir::Value add_res, mul_res;
            if (elem_type.isF32()) {
                mul_res = mlir::arith::MulFOp::create(b, loc, args[0], args[1]);
                add_res = mlir::arith::AddFOp::create(b, loc, args[2], mul_res);
            } else {
                mul_res = mlir::arith::MulIOp::create(b, loc, args[0], args[1]);
                add_res = mlir::arith::AddIOp::create(b, loc, args[2], mul_res);
            }

            mlir::linalg::YieldOp::create(b, loc, add_res);
        };

        mlir::linalg::GenericOp generic = mlir::linalg::GenericOp::create(
            builder, location, outType, mlir::ValueRange{input1, input2}, init, maps, iters, body);
        valueTable_insertion(op.outputs_[0], generic.getResult(0));
    }

    void gen_gemm(const Labs::Op &op) {
        mlir::Value input1 = valueTable.at(op.inputs_[0]);
        mlir::Value input2 = valueTable.at(op.inputs_[1]);
        mlir::RankedTensorType outType = get_type(op.outputs_[0]);
        mlir::Type elem_type = outType.getElementType();
        check_supported_type(elem_type);

        const Labs::Gemm &gemm = static_cast<const Labs::Gemm &>(op);
        mlir::Value init = gen_gemm_init(op);
        mlir::SmallVector<mlir::utils::IteratorType> iters(2, mlir::utils::IteratorType::parallel);
        iters.push_back(mlir::utils::IteratorType::reduction);

        auto expr_dim = [this](std::size_t rank) { return builder.getAffineDimExpr(rank); };
        mlir::AffineExpr d0 = expr_dim(0);
        mlir::AffineExpr d1 = expr_dim(1);
        mlir::AffineExpr d2 = expr_dim(2);
        mlir::SmallVector<mlir::AffineExpr> input1_exprs, input2_exprs, out_exprs;
        out_exprs = {d0, d1};
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

        mlir::AffineMap input1_map = mlir::AffineMap::get(3, 0, input1_exprs, builder.getContext());
        mlir::AffineMap input2_map = mlir::AffineMap::get(3, 0, input2_exprs, builder.getContext());
        mlir::AffineMap out_map = mlir::AffineMap::get(3, 0, out_exprs, builder.getContext());
        mlir::SmallVector<mlir::AffineMap> maps = {input1_map, input2_map, out_map};

        mlir::Location location = convert_str_to_loc(gemm.outputs_[0]->name_);
        mlir::Value alpha = mlir::arith::ConstantOp::create(builder, location,
                                                            builder.getF32FloatAttr(gemm.alpha_));

        auto body = [alpha, elem_type](mlir::OpBuilder &b, mlir::Location loc,
                                       mlir::ValueRange args) {
            mlir::Value ab_mul, alpha_ab_mul, res;
            if (elem_type.isF32()) {
                ab_mul = mlir::arith::MulFOp::create(b, loc, args[0], args[1]);
                alpha_ab_mul = mlir::arith::MulFOp::create(b, loc, alpha, ab_mul);
                res = mlir::arith::AddFOp::create(b, loc, args[2], alpha_ab_mul);
            } else {
                ab_mul = mlir::arith::MulIOp::create(b, loc, args[0], args[1]);
                alpha_ab_mul = mlir::arith::MulIOp::create(b, loc, alpha, ab_mul);
                res = mlir::arith::AddIOp::create(b, loc, args[2], alpha_ab_mul);
            }

            mlir::linalg::YieldOp::create(b, loc, res);
        };

        mlir::linalg::GenericOp generic = mlir::linalg::GenericOp::create(
            builder, location, outType, mlir::ValueRange{input1, input2}, init, maps, iters, body);
        valueTable_insertion(gemm.outputs_[0], generic.getResult(0));
    }

    void gen_conv(const Labs::Conv &conv) {
        mlir::Value X = gen_conv_init(conv);
        mlir::Value K = valueTable.at(conv.inputs_[1]);
        int64_t G = conv.group_;
        mlir::Location location = convert_str_to_loc(conv.outputs_[0]->name_);
        mlir::Type elem_type = mlir::getElementTypeOrSelf(X);
        check_supported_type(elem_type);

        mlir::RankedTensorType X_padded_type = llvm::cast<mlir::RankedTensorType>(X.getType());
        auto [split_shape_X, split_groups_X] = split_dim(X_padded_type.getShape(), G, 1);
        mlir::RankedTensorType X_split_type = mlir::RankedTensorType::get(split_shape_X, elem_type);
        mlir::Value X_new =
            mlir::tensor::ExpandShapeOp::create(builder, location, X_split_type, X, split_groups_X);

        auto [split_shape_K, split_groups_K] = split_dim(conv.inputs_[1]->shape_, G, 0);
        mlir::RankedTensorType K_split_type = mlir::RankedTensorType::get(split_shape_K, elem_type);
        mlir::Value K_new =
            mlir::tensor::ExpandShapeOp::create(builder, location, K_split_type, K, split_groups_K);

        auto [split_shape_Y, split_groups_y] = split_dim(conv.outputs_[0]->shape_, G, 1);
        mlir::RankedTensorType split_Y_type = mlir::RankedTensorType::get(split_shape_Y, elem_type);
        mlir::Value zero = mlir::arith::getZeroConstant(builder, location, elem_type);
        mlir::Value empty =
            mlir::tensor::EmptyOp::create(builder, location, split_shape_Y, elem_type);
        mlir::Value init =
            mlir::linalg::FillOp::create(builder, location, zero, empty).getResult(0);

        std::size_t dims_amount = conv.inputs_[0]->shape_.size() - 2;
        mlir::SmallVector<mlir::utils::IteratorType> iters;
        iters.append(dims_amount + 3, mlir::utils::IteratorType::parallel);
        iters.append(dims_amount + 1, mlir::utils::IteratorType::reduction);

        mlir::SmallVector<mlir::AffineExpr> out_exprs;
        for (std::size_t i = 0; i < 3 + dims_amount; ++i) {
            out_exprs.push_back(builder.getAffineDimExpr(i));
        }

        mlir::SmallVector<mlir::AffineExpr> K_exprs;
        K_exprs.push_back(builder.getAffineDimExpr(1));
        K_exprs.push_back(builder.getAffineDimExpr(2));
        for (std::size_t i = dims_amount + 3; i < 2 * dims_amount + 4; ++i) {
            K_exprs.push_back(builder.getAffineDimExpr(i));
        }

        mlir::SmallVector<mlir::AffineExpr> X_exprs;
        X_exprs.push_back(builder.getAffineDimExpr(0));
        X_exprs.push_back(builder.getAffineDimExpr(1));
        X_exprs.push_back(builder.getAffineDimExpr(dims_amount + 3));
        for (std::size_t i = 0; i < dims_amount; ++i) {
            mlir::AffineExpr cur =
                builder.getAffineDimExpr(i + 3) * conv.strides_[i] +
                builder.getAffineDimExpr(i + dims_amount + 4) * conv.dilations_[i];
            X_exprs.push_back(cur);
        }

        mlir::AffineMap out_map, K_map, X_map;
        out_map = mlir::AffineMap::get(2 * dims_amount + 4, 0, out_exprs, builder.getContext());
        K_map = mlir::AffineMap::get(2 * dims_amount + 4, 0, K_exprs, builder.getContext());
        X_map = mlir::AffineMap::get(2 * dims_amount + 4, 0, X_exprs, builder.getContext());
        mlir::SmallVector<mlir::AffineMap> maps = {X_map, K_map, out_map};

        auto body = [elem_type](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange args) {
            mlir::Value mul_res, res;
            if (elem_type.isF32()) {
                mul_res = mlir::arith::MulFOp::create(b, loc, args[0], args[1]);
                res = mlir::arith::AddFOp::create(b, loc, args[2], mul_res);
            } else {
                mul_res = mlir::arith::MulIOp::create(b, loc, args[0], args[1]);
                res = mlir::arith::AddIOp::create(b, loc, args[2], mul_res);
            }

            mlir::linalg::YieldOp::create(b, loc, res);
        };

        mlir::RankedTensorType Y_type = get_type(conv.outputs_[0]);
        mlir::linalg::GenericOp generic = mlir::linalg::GenericOp::create(
            builder, location, split_Y_type, {X_new, K_new}, init, maps, iters, body);
        mlir::Value Y = mlir::tensor::CollapseShapeOp::create(builder, location, Y_type,
                                                              generic.getResult(0), split_groups_y);
        mlir::Value table_insertion = Y;

        //------------------------------------------------------------------------------------------
        if (conv.inputs_.size() == 3) {
            mlir::Value bias = valueTable.at(conv.inputs_[2]);
            mlir::Value bias_init =
                mlir::tensor::EmptyOp::create(builder, location, Y_type.getShape(), elem_type);
            mlir::SmallVector<mlir::utils::IteratorType> bias_iters(
                dims_amount + 2, mlir::utils::IteratorType::parallel);
            mlir::AffineExpr bias_exprs = builder.getAffineDimExpr(1);

            mlir::AffineMap bias_map =
                mlir::AffineMap::get(2 + dims_amount, 0, bias_exprs, builder.getContext());
            mlir::AffineMap Y_map = builder.getMultiDimIdentityMap(2 + dims_amount);
            mlir::AffineMap bias_out_map = Y_map;
            mlir::SmallVector<mlir::AffineMap> bias_maps = {Y_map, bias_map, bias_out_map};

            auto bias_body = [elem_type](mlir::OpBuilder &b, mlir::Location loc,
                                         mlir::ValueRange args) {
                mlir::Value res;
                if (elem_type.isF32()) {
                    res = mlir::arith::AddFOp::create(b, loc, args[0], args[1]);
                } else {
                    res = mlir::arith::AddIOp::create(b, loc, args[0], args[1]);
                }

                mlir::linalg::YieldOp::create(b, loc, res);
            };

            mlir::linalg::GenericOp bias_generic = mlir::linalg::GenericOp::create(
                builder, location, Y_type, {Y, bias}, bias_init, bias_maps, bias_iters, bias_body);
            table_insertion = bias_generic.getResult(0);
        }

        valueTable_insertion(conv.outputs_[0], table_insertion);
    }

    void gen_elementwise(const Labs::Op &op, BodyBuilder lambda) {
        mlir::RankedTensorType outType = get_type(op.outputs_[0]);
        mlir::Type elem_type = outType.getElementType();
        check_supported_type(elem_type);

        mlir::SmallVector<mlir::Value> inputs;
        mlir::SmallVector<mlir::AffineMap> maps;
        std::size_t rank = op.outputs_[0]->shape_.size();

        for (const auto *input_ptr : op.inputs_) {
            mlir::Value val_i = valueTable.at(input_ptr);

            maps.push_back(get_broadcast_map(input_ptr->shape_, op.outputs_[0]->shape_));
            inputs.push_back(val_i);
        }
        maps.push_back(builder.getMultiDimIdentityMap(rank));

        mlir::Location location = convert_str_to_loc(op.outputs_[0]->name_);
        mlir::Value init =
            mlir::tensor::EmptyOp::create(builder, location, outType.getShape(), elem_type);
        mlir::SmallVector<mlir::utils::IteratorType> iters(rank,
                                                           mlir::utils::IteratorType::parallel);

        mlir::linalg::GenericOp generic = mlir::linalg::GenericOp::create(
            builder, location, outType, inputs, init, maps, iters, lambda);
        valueTable_insertion(op.outputs_[0], generic.getResult(0));
    }

    // Prepares mlir::Value equal to the bias of the desired size
    mlir::Value gen_gemm_init(const Labs::Op &op) {
        mlir::RankedTensorType outType = get_type(op.outputs_[0]);
        mlir::Type elem_type = outType.getElementType();
        mlir::Location location = convert_str_to_loc(op.outputs_[0]->name_);
        mlir::Value bias = valueTable.at(op.inputs_[2]);
        mlir::Value init =
            mlir::tensor::EmptyOp::create(builder, location, outType.getShape(), elem_type);

        if (op.inputs_.size() == 2) {
            return init;
        }

        mlir::AffineMap bias_map = get_broadcast_map(op.inputs_[2]->shape_, op.outputs_[0]->shape_);
        mlir::AffineMap output_map = builder.getMultiDimIdentityMap(op.outputs_[0]->shape_.size());
        mlir::SmallVector<mlir::AffineMap> maps = {bias_map, output_map};
        mlir::SmallVector<mlir::utils::IteratorType> iters(op.outputs_[0]->shape_.size(),
                                                           mlir::utils::IteratorType::parallel);

        const Labs::Gemm &gemm = static_cast<const Labs::Gemm &>(op);
        mlir::Value beta =
            mlir::arith::ConstantOp::create(builder, location, builder.getF32FloatAttr(gemm.beta_));
        auto body = [beta, elem_type](mlir::OpBuilder &b, mlir::Location loc,
                                      mlir::ValueRange args) {
            mlir::Value res;
            if (elem_type.isF32()) {
                res = mlir::arith::MulFOp::create(b, loc, beta, args[0]);
            } else {
                res = mlir::arith::MulIOp::create(b, loc, beta, args[0]);
            }

            mlir::linalg::YieldOp::create(b, loc, res);
        };

        mlir::linalg::GenericOp generic = mlir::linalg::GenericOp::create(
            builder, location, outType, bias, init, maps, iters, body);
        return generic.getResult(0);
    }

    // prepares mlir::Value X: if necessary, performs padding
    mlir::Value gen_conv_init(const Labs::Conv &conv) {
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

        mlir::RankedTensorType temp_type = get_type(conv.inputs_[0]);
        mlir::Type elem_type = temp_type.getElementType();
        mlir::RankedTensorType padded_type = mlir::RankedTensorType::get(padded_shape, elem_type);
        mlir::Location location = convert_str_to_loc(conv.outputs_[0]->name_);
        mlir::Value x = valueTable.at(conv.inputs_[0]);
        auto low = mlir::getAsIndexOpFoldResult(builder.getContext(), low_ints);
        auto high = mlir::getAsIndexOpFoldResult(builder.getContext(), high_ints);
        mlir::Value pad_value;
        pad_value = mlir::arith::getZeroConstant(builder, location, elem_type);

        mlir::Value result =
            mlir::tensor::PadOp::create(builder, location, padded_type, x, low, high, pad_value);
        return result;
    }

    // Full broadcast map over output.size() loop dims.
    mlir::AffineMap get_broadcast_map(mlir::ArrayRef<int64_t> input,
                                      mlir::ArrayRef<int64_t> output) {
        if (input == output)
            return builder.getMultiDimIdentityMap(input.size());

        mlir::SmallVector<mlir::AffineExpr> exprs = get_broadcast_exprs(input, output);
        return mlir::AffineMap::get(output.size(), 0, exprs, builder.getContext());
    }

    // Broadcast index expressions for `input` aligned to `output` (NumPy-style):
    // missing leading dims are skipped, size-1 dims map to 0.
    // For MatMul pass only batch dims!
    mlir::SmallVector<mlir::AffineExpr> get_broadcast_exprs(mlir::ArrayRef<int64_t> input,
                                                            mlir::ArrayRef<int64_t> output) {
        mlir::SmallVector<mlir::AffineExpr> input_affine_expr;
        std::size_t size_diff = output.size() - input.size();

        for (std::size_t i = 0; i < input.size(); ++i) {
            bool dim_equal_unit = (input[i] == 1);
            mlir::AffineExpr cur_expr = (dim_equal_unit) ? builder.getAffineConstantExpr(0)
                                                         : builder.getAffineDimExpr(size_diff + i);
            input_affine_expr.push_back(cur_expr);
        }
        return input_affine_expr;
    }

    SplitResult split_dim(const std::vector<int64_t> &shape, const std::size_t G,
                          const std::size_t pos) {
        std::vector<int64_t> split_shape = shape;
        split_shape[pos] /= G;
        split_shape.insert(split_shape.begin() + pos, G);

        mlir::SmallVector<mlir::ReassociationIndices> groups;
        for (int64_t i = 0; i < shape.size(); ++i) {
            if (i < pos) {
                groups.push_back({i});
            } else if (i == pos) {
                groups.push_back({i, i + 1});
            } else {
                groups.push_back({i + 1});
            }
        }
        return {split_shape, groups};
    }

}; // class MLIRGenImpl

mlir::SmallVector<int64_t> convert_shape_to_mlir(const std::vector<int64_t> &val_shape) {
    mlir::SmallVector<int64_t> shape(val_shape.begin(), val_shape.end());
    int64_t absent_dim = -1;

    std::replace(shape.begin(), shape.end(), absent_dim, mlir::ShapedType::kDynamic);
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

void check_supported_type(mlir::Type elem_type) {
    if (!elem_type.isF32() && !elem_type.isInteger(64)) {
        throw std::runtime_error(
            llvm::formatv("Unsupported type of elements: {0}", elem_type).str());
    }
}
} // namespace

namespace Labs {
mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context, const ONNX_Graph &graph) {
    return MLIRGenImpl(context).mlirGen(graph);
}
} // namespace Labs