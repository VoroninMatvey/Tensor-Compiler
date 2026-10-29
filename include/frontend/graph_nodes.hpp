#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace Labs {

class Value;

class GraphNode {
  public:
    virtual ~GraphNode() = default;
}; // <-- class GraphNode

class Op : public GraphNode {
  public:
    std::vector<Value *> inputs_;
    std::vector<Value *> outputs_;
    std::string op_type_;

}; // <-- class Op

class Value : public GraphNode {
  public:
    enum class DataStatus { Present, Absent };
    enum class DataType { Int64, Float32 };

    DataType type_ = DataType::Float32;
    std::string name_; // required strictly during import
    std::vector<int64_t> shape_;

    DataStatus status_ = DataStatus::Absent;
    std::vector<std::byte> raw_data_;

    Op *producer_ = nullptr;
    std::vector<Op *> consumers_;

}; // <-- class Value

using DataType = Value::DataType;
using DataStatus = Value::DataStatus;

class Add : public Op {};
class Mul : public Op {};
class Relu : public Op {};
class MatMul : public Op {};

class Gemm : public Op {
  public:
    float alpha_ = 1.0f;
    float beta_ = 1.0f;
    bool need_transA_ = false;
    bool need_transB_ = false;
};

class Conv : public Op {
  public:
    std::string auto_pad_ = "NOTSET";
    std::vector<int64_t> dilations_;
    int64_t group_ = 1;
    std::vector<int64_t> kernel_shape_;
    std::vector<int64_t> pads_;
    std::vector<int64_t> strides_;
};

} // namespace Labs