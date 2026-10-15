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
#include <functional>
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

  public:
    MLIRGenImpl(mlir::MLIRContext &context) : builder(&context) {}

    mlir::OwningOpRef<mlir::ModuleOp> mlirGen(const Labs::ONNX_Graph &my_graph) {
        mlir::OwningOpRef owner = mlir::ModuleOp::create(builder.getUnknownLoc());
        theModule = owner.get();

        mlir::func::FuncOp main = create_main(my_graph.inputs, my_graph.outputs);
        builder.setInsertionPointToStart(&main.front());

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
        for (size_t i = 0; i < inputs.size(); ++i)
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

    mlir::Location convert_str_to_loc(const std::string &op_name) {
        mlir::StringAttr name = builder.getStringAttr(op_name);
        return mlir::NameLoc::get(name);
    }

    void valueTable_insertion(const Labs::Value *graph_val, mlir::Value mlir_val) {
        if (!valueTable.emplace(graph_val, mlir_val).second) {
            throw std::runtime_error("Value " + graph_val->name_ + " was defined twice");
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

        size_t rank = op.outputs_[0]->shape_.size();
        mlir::Location location = convert_str_to_loc(op.outputs_[0]->name_);
        mlir::Value zero = mlir::arith::getZeroConstant(builder, location, elem_type);
        mlir::Value empty =
            mlir::tensor::EmptyOp::create(builder, location, outType.getShape(), elem_type);
        mlir::Value init =
            mlir::linalg::FillOp::create(builder, location, outType, zero, empty).getResult(0);

        mlir::SmallVector<mlir::utils::IteratorType> iters(rank,
                                                           mlir::utils::IteratorType::parallel);
        iters.push_back(mlir::utils::IteratorType::reduction);

        bool a_is_vec = (op.inputs_[0]->shape_.size() == 0);
        bool b_is_vec = (op.inputs_[1]->shape_.size() == 0);
        size_t out_offset = (a_is_vec ? 0 : 1) + (b_is_vec ? 0 : 1);
        mlir::AffineExpr di = builder.getAffineDimExpr(rank - 2);
        mlir::AffineExpr dj = builder.getAffineDimExpr(rank - 1);
        mlir::AffineExpr dk = builder.getAffineDimExpr(rank);

        mlir::SmallVector<mlir::AffineExpr> input1_expr;
        if (a_is_vec) {
            input1_expr.push_back(dk);
        } else {
            input1_expr =
                get_broadcast_exprs(op.inputs_[0]->shape_, op.outputs_[0]->shape_, 2, out_offset);
            input1_expr.push_back(di);
            input1_expr.push_back(dk);
        }

        mlir::SmallVector<mlir::AffineExpr> input2_expr;
        if (a_is_vec) {
            input2_expr.push_back(dk);
        } else {
            input2_expr =
                get_broadcast_exprs(op.inputs_[1]->shape_, op.outputs_[0]->shape_, 2, out_offset);
            input2_expr.push_back(dk);
            input2_expr.push_back(dj);
        }

        mlir::SmallVector<mlir::AffineExpr> output_expr;
        for (size_t i = 0; i < rank; ++i) {
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

    void gen_elementwise(const Labs::Op &op, BodyBuilder lambda) {
        mlir::RankedTensorType outType = get_type(op.outputs_[0]);
        mlir::Type elem_type = outType.getElementType();
        check_supported_type(elem_type);

        mlir::SmallVector<mlir::Value> inputs;
        mlir::SmallVector<mlir::AffineMap> maps;
        size_t rank = op.outputs_[0]->shape_.size();

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

    // Builds the indexing map of an input operand in the iteration space of the output,
    // following ONNX broadcast rules: missing leading dims are skipped,
    // dims of size 1 are mapped to constant 0, the rest to the matching loop dim.
    mlir::AffineMap get_broadcast_map(mlir::ArrayRef<int64_t> input,
                                      mlir::ArrayRef<int64_t> output) {
        if (input == output)
            return builder.getMultiDimIdentityMap(input.size());

        mlir::SmallVector<mlir::AffineExpr> input_affine_expr = get_broadcast_exprs(input, output);
        return mlir::AffineMap::get(output.size(), 0, input_affine_expr, builder.getContext());
    }

    // Same rules as get_broadcast_map, but returns the list of expressions
    // and ignores the last `offset` dims of the input (e.g. offset = 2 for
    // the matrix dims of MatMul, so only batch dims are broadcast).
    mlir::SmallVector<mlir::AffineExpr> get_broadcast_exprs(mlir::ArrayRef<int64_t> input,
                                                            mlir::ArrayRef<int64_t> output,
                                                            size_t in_offset = 0,
                                                            size_t out_offset = 0) {
        mlir::SmallVector<mlir::AffineExpr> input_affine_expr;
        size_t in_batch = input.size() - in_offset;
        size_t out_batch = output.size() - out_offset;
        size_t size_diff = out_batch - in_batch;

        for (size_t i = 0; i < in_batch; ++i) {
            bool dim_equal_unit = (input[i] == 1);
            mlir::AffineExpr cur_expr = (dim_equal_unit) ? builder.getAffineConstantExpr(0)
                                                         : builder.getAffineDimExpr(size_diff + i);
            input_affine_expr.push_back(cur_expr);
        }
        return input_affine_expr;
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