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
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
llvm::SmallVector<int64_t> convert_shape_to_mlir(const std::vector<int64_t> &val_shape);

class MLIRGenImpl {
  public:
    MLIRGenImpl(mlir::MLIRContext &context) : builder(&context) {}

    mlir::OwningOpRef<mlir::ModuleOp> mlirGen(const Labs::ONNX_Graph &my_graph) {
        mlir::OwningOpRef owner = mlir::ModuleOp::create(builder.getUnknownLoc());
        theModule = owner.get();

        mlir::func::FuncOp main = create_main(my_graph.inputs, my_graph.outputs);
        builder.setInsertionPointToStart(&main.front());

        for (const auto &op_ref : my_graph.ops_) {
            Labs::Op *op = op_ref.get();
            const std::string &op_type = op->op_type_;

            if (op_type == "Relu") {
                gen_relu(static_cast<Labs::Relu &>(*op));
            } else if (op_type == "Add") {
                gen_add(*op);
            } else {
                throw std::runtime_error("unsupported gen operation now: " + op_type);
            }
        }

        llvm::SmallVector<mlir::Value> results;
        for (const auto *out_val : my_graph.outputs) {
            results.push_back(valueTable.at(out_val));
        }
        mlir::func::ReturnOp::create(builder, builder.getUnknownLoc(), results);

        return owner;
    }

    mlir::func::FuncOp create_main(const std::vector<Labs::Value *> &inputs,
                                   const std::vector<Labs::Value *> &outputs) {
        mlir::Location location = convert_str_to_loc("onnx_graph");

        llvm::SmallVector<mlir::Type> argsTypes = get_types(inputs);
        llvm::SmallVector<mlir::Type> resTypes = get_types(outputs);

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

        llvm::SmallVector<int64_t> mlir_shape = convert_shape_to_mlir(value->shape_);
        return mlir::RankedTensorType::get(mlir_shape, mlir_type);
    }

    llvm::SmallVector<mlir::Type> get_types(const std::vector<Labs::Value *> &vec) {
        llvm::SmallVector<mlir::Type> vecTypes;
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

    void gen_relu(const Labs::Relu &op) {
        mlir::Value input = valueTable.at(op.inputs_[0]);
        mlir::RankedTensorType outType = get_type(op.outputs_[0]);
        mlir::Type elem_type = outType.getElementType();

        if (!elem_type.isF32() && !elem_type.isInteger(64)) {
            throw std::runtime_error(
                llvm::formatv("Unsupported type of elements: {0}", elem_type).str());
        }

        mlir::Location location = convert_str_to_loc(op.outputs_[0]->name_);
        mlir::Value init =
            mlir::tensor::EmptyOp::create(builder, location, outType.getShape(), elem_type);

        int64_t rank = outType.getRank();
        mlir::AffineMap id = builder.getMultiDimIdentityMap(rank);
        llvm::SmallVector<mlir::AffineMap> maps = {id, id};
        llvm::SmallVector<mlir::utils::IteratorType> iters(rank,
                                                           mlir::utils::IteratorType::parallel);

        auto body = [elem_type](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange args) {
            mlir::Value zero = mlir::arith::getZeroConstant(b, loc, elem_type);

            mlir::Value r;
            if (elem_type.isF32()) {
                r = mlir::arith::MaximumFOp::create(b, loc, args[0], zero);
            } else {
                r = mlir::arith::MaxSIOp::create(b, loc, args[0], zero);
            }

            mlir::linalg::YieldOp::create(b, loc, r);
        };

        mlir::linalg::GenericOp generic = mlir::linalg::GenericOp::create(
            builder, location, outType, input, init, maps, iters, body);
        valueTable_insertion(op.outputs_[0], generic.getResult(0));
    }

    void gen_add(const Labs::Op &op) {
        mlir::Value input1 = valueTable.at(op.inputs_[0]);
        mlir::Value input2 = valueTable.at(op.inputs_[1]);

        mlir::RankedTensorType outType = get_type(op.outputs_[0]);
        mlir::Type elem_type = outType.getElementType();

        if (!elem_type.isF32() && !elem_type.isInteger(64)) {
            throw std::runtime_error(
                llvm::formatv("Unsupported type of elements: {0}", elem_type).str());
        }

        int64_t rank = outType.getRank();
        mlir::AffineMap lhs_map = get_broadcast_map(op.inputs_[0]->shape_, op.outputs_[0]->shape_);
        mlir::AffineMap rhs_map = get_broadcast_map(op.inputs_[1]->shape_, op.outputs_[0]->shape_);
        mlir::AffineMap out_map = builder.getMultiDimIdentityMap(rank);
        mlir::SmallVector<mlir::AffineMap, 3> maps = {lhs_map, rhs_map, out_map};
        mlir::SmallVector<mlir::utils::IteratorType> iters(rank,
                                                           mlir::utils::IteratorType::parallel);

        mlir::Location location = convert_str_to_loc(op.outputs_[0]->name_);
        mlir::Value init =
            mlir::tensor::EmptyOp::create(builder, location, outType.getShape(), elem_type);

        auto body = [elem_type](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange args) {
            mlir::Value res;
            if (elem_type.isF32()) {
                res = mlir::arith::AddFOp::create(b, loc, args[0], args[1]);
            } else {
                res = mlir::arith::AddIOp::create(b, loc, args[0], args[1]);
            }

            mlir::linalg::YieldOp::create(b, loc, res);
        };

        mlir::linalg::GenericOp generic = mlir::linalg::GenericOp::create(
            builder, location, outType, mlir::ValueRange{input1, input2}, init, maps, iters, body);
        valueTable_insertion(op.outputs_[0], generic.getResult(0));
    }

    mlir::AffineMap get_broadcast_map(mlir::ArrayRef<int64_t> input,
                                      mlir::ArrayRef<int64_t> output) {
        if (input == output)
            return builder.getMultiDimIdentityMap(input.size());

        llvm::SmallVector<mlir::AffineExpr> input_affine_expr;
        size_t offset = output.size() - input.size();

        for (size_t i = 0; i < input.size(); ++i) {
            bool dim_equal_unit = (input[i] == 1);
            mlir::AffineExpr cur_expr = (dim_equal_unit) ? builder.getAffineConstantExpr(0)
                                                         : builder.getAffineDimExpr(offset + i);
            input_affine_expr.push_back(cur_expr);
        }
        return mlir::AffineMap::get(output.size(), 0, input_affine_expr, builder.getContext());
    }

}; // class MLIRGenImpl

llvm::SmallVector<int64_t> convert_shape_to_mlir(const std::vector<int64_t> &val_shape) {
    llvm::SmallVector<int64_t> shape(val_shape.begin(), val_shape.end());
    int64_t absent_dim = -1;

    std::replace(shape.begin(), shape.end(), absent_dim, mlir::ShapedType::kDynamic);
    return shape;
}
} // namespace

namespace Labs {
mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context, const ONNX_Graph &graph) {
    return MLIRGenImpl(context).mlirGen(graph);
}
} // namespace Labs