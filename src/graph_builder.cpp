#include <cstdint>
#include <format>
#include <graph_builder.hpp>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace Labs {

namespace {

struct TensorStyle {
    std::string_view bg_color_;
    std::string_view color_;
}; // struct TensorStule

namespace TensorColors {

constexpr TensorStyle Input = {"#E8F5E9", "#2E7D32"};
constexpr TensorStyle Output = {"#FFEBEE", "#C62828"};
constexpr TensorStyle Weight = {"#FFF8E1", "#F57F17"};
constexpr TensorStyle Other = {"#E3F2FD", "#1565C0"};

} // namespace TensorColors

void printOp(const Op *op_ptr, std::string &dot, int id);
std::string shape_to_string(const std::vector<int64_t> &shape);

} // namespace

std::string build_dot(const ONNX_Graph &my_graph, const std::string &fontname,
                      const std::string &edge_color, const double penwidth) {

    std::unordered_set<std::string> inserted_value;
    std::string dot;

    // clang-format off
    dot += std::format(R"DOT(
    strict digraph ONNX_graph {
    
        node = [fontname = "{}"]; 
        edge = [color = "{}", pendwith = {}];  

    )DOT", fontname, edge_color, penwidth);
    // clang-format on

    //--------------------- separate function
    for (const auto *input_ptr : my_graph.inputs) {
        inserted_value.insert(input_ptr->name_);
        printTensor(input_ptr, dot, TensorColors::Input);
    }

    for (const auto *output_ptr : my_graph.outputs) {
        inserted_value.insert(output_ptr->name_);
        printTensor(output_ptr, dot, TensorColors::Output);
    }

    for (const auto &value : my_graph.values_) {
        if (inserted_value.find(value->name_) != inserted_value.end())
            continue;

        const Value *value_ptr = value.get();
        inserted_value.insert(value_ptr->name_);
        TensorStyle style = (value_ptr->status_ == DataStatus::Present) ? TensorColors::Weight
                                                                        : TensorColors::Other;
        printTensor(value_ptr, dot, style);
    }
    //---------------------

    std::unordered_map<std::string, int> op_id_map = init_op_id();
    for (const auto &op : my_graph.ops_) {
        const Op *op_ptr = op.get();
        int op_id = op_id_map[op->op_type_]++;
        std::string op_name = op->op_type_ + "_" + std::to_string(op_id);

        printOp(dot, op_name);
    }

    return "ada";
}

namespace {

std::unordered_map<std::string, int> init_op_id() {
    std::vector<std::string> vec = {"Add", "Mul", "Relu", "MatMul", "Conv", "Gemm"};
    std::unordered_map<std::string, int> op_id_map;

    for (int i = 0; i < 5; ++i) {
        op_id_map.insert({vec[i], 0});
    }

    return op_id_map;
}

// clang-format off
void printOp(std::string &dot, std::string_view op_name) {
    dot += std::format(R"DOT(
        node [shape = ellipse, style = solid]
        "{}"
    )DOT", op_name);
}


void printTensor(const Value* val_ptr, std::string& dot, const TensorStyle& style) {
    dot += std::format(R"DOT(
        "{1}" [shape = none, label=<
            <TABLE BORDER = "1" CELLBORDER = "1" CELLSPACING = "0" CELLPADDING = "4" BGCOLOR = "{2}" COLOR = "{3}">
                <TR><TD HEIGHT="40">{1}</TD></TR>
                <TR><TD HEIGHT="20">{4}</TD></TR>
            </TABLE>
        >];
    )DOT", val_ptr->name_, style.bg_color_, style.color_, shape_to_string(val_ptr->shape_));
}

void printEdges(const Op* op_ptr, std::string& dot, std::string_view op_name) {
    dot += std::format(R"DOT(
        "{}" -> "{}";
        // inputs of op
    )DOT", op_name, op_ptr->outputs_[1]->name_);
}
// clang-format on

std::string shape_to_string(const std::vector<int64_t> &shape) {
    std::string shape_str;
    shape_str += "[";

    for (const auto &dim : shape) {
        shape_str += std::to_string(dim);
        shape_str += ", ";
    }

    std::size_t size = shape_str.size();
    if (size > 2)
        shape_str.erase(size - 2, 2);

    shape_str += "]";
    return shape_str;
}

} // namespace
} // namespace Labs