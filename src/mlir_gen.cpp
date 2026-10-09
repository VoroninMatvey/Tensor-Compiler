#include "mlir_gen.hpp"
#include "graph.hpp"
#include "graph_nodes.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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
        builder.setInsertionPointToEnd(&main.front());
        mlir::func::ReturnOp::create(builder, builder.getUnknownLoc());

        return owner;
    }

    mlir::func::FuncOp create_main(const std::vector<Labs::Value *> &inputs,
                                   const std::vector<Labs::Value *> &outputs) {
        mlir::StringAttr name = builder.getStringAttr("onnx_graph");
        mlir::Location location = mlir::NameLoc::get(name);

        llvm::SmallVector<mlir::Type> argsTypes = get_types(inputs);
        llvm::SmallVector<mlir::Type> resTypes = get_types(outputs);

        // temporary output stub
        mlir::FunctionType funcType = builder.getFunctionType(argsTypes, {});
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

    mlir::Type get_type(const Labs::Value *value) {
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

    void valueTable_insertion(const Labs::Value *graph_val, mlir::Value mlir_val) {
        if (!valueTable.emplace(graph_val, mlir_val).second) {
            throw std::runtime_error("Value " + graph_val->name_ + " was define twice");
        }
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