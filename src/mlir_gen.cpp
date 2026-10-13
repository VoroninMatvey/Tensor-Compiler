#include "mlir_gen.hpp"
#include "graph.hpp"
#include "graph_nodes.hpp"
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
            std::string op_type = op->op_type_;

            if (op_type == "Relu") {
                gen_relu(static_cast<Labs::Relu &>(*op));
            } else {
                throw std::runtime_error("unsupported gen operation now");
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

        // temporary output stub
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

    mlir::Location convert_str_to_loc(std::string op_name) {
        mlir::StringAttr name = builder.getStringAttr(op_name);
        return mlir::NameLoc::get(name);
    }

    void valueTable_insertion(const Labs::Value *graph_val, mlir::Value mlir_val) {
        if (!valueTable.emplace(graph_val, mlir_val).second) {
            throw std::runtime_error("Value " + graph_val->name_ + " was define twice");
        }
    }

    void gen_relu(const Labs::Relu &op) {
        mlir::Value input = valueTable.at(op.inputs_[0]);
        mlir::RankedTensorType outType = get_type(op.outputs_[0]);
        mlir::Location location = convert_str_to_loc(op.outputs_[0]->name_);
        mlir::Value init = mlir::tensor::EmptyOp::create(builder, location, outType.getShape(),
                                                         outType.getElementType());

        int64_t rank = outType.getRank();
        mlir::AffineMap id = builder.getMultiDimIdentityMap(rank);
        llvm::SmallVector<mlir::AffineMap> maps = {id, id};
        llvm::SmallVector<mlir::utils::IteratorType> iters(rank,
                                                           mlir::utils::IteratorType::parallel);

        auto body = [](mlir::OpBuilder &b, mlir::Location loc, mlir::ValueRange args) {
            mlir::Value zero = mlir::arith::getZeroConstant(b, loc, b.getF32Type());
            mlir::Value r = mlir::arith::MaximumFOp::create(b, loc, args[0], zero);
            mlir::linalg::YieldOp::create(b, loc, r);
        };

        mlir::linalg::GenericOp generic = mlir::linalg::GenericOp::create(
            builder, location, outType, input, init, maps, iters, body);
        valueTable_insertion(op.outputs_[0], generic.getResult(0));
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