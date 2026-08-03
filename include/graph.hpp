#pragma once
#include "graph_nodes.hpp"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace Labs {

class ONNX_Graph {
  public:
    std::vector<std::unique_ptr<Op>> ops_;
    std::vector<std::unique_ptr<Value>> values_;

    std::vector<Value *> inputs;
    std::vector<Value *> outputs;
    std::unordered_map<std::string, Value *> name_to_value_;

    Value *addValue(data_type type, std::string name, std::vector<int64_t> shape,
                    data_status status) {
        std::unique_ptr<Value> val_uniq = std::make_unique<Value>();
        Value *val_ptr = val_uniq.get();
        val_ptr->type_ = type;
        val_ptr->name_ = name;
        val_ptr->shape_ = shape;
        val_ptr->status_ = status;

        name_to_value_.insert({name, val_ptr});
        values_.push_back(std::move(val_uniq));
        return val_ptr;
    }

    template <typename OpT> OpT *addOp() {
        std::unique_ptr<OpT> op_uniq = std::make_unique<OpT>();
        OpT *op_ptr = op_uniq.get();

        ops_.push_back(std::move(op_uniq));
        return op_ptr;
    }
}; // class ONNX_Graph
} // namespace Labs