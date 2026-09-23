#include "graph_builder.hpp"
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <stdexcept>
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

void printOp(std::string &dot, std::string_view op_name);
void printTensor(const Value *val_ptr, std::string &dot, const TensorStyle &style);
void printEdges(const Op *op_ptr, std::string &dot, std::string_view op_name,
                const std::string &input_str);

void print_ops(const ONNX_Graph &my_graph, std::string &dot);
void print_values(const ONNX_Graph &my_graph, std::string &dot);

std::string shape_to_string(const std::vector<int64_t> &shape);
std::string op_input_to_string(const std::vector<Value *> &input, std::string_view op_name);

} // namespace

std::string build_dot(const ONNX_Graph &my_graph, const std::string &fontname,
                      const std::string &edge_color, const double penwidth) {

    std::string dot;

    // clang-format off
    dot += std::format(R"DOT(
    strict digraph ONNX_graph {{
    
        node [fontname = "{}"]; 
        edge [color = "{}", penwidth = {}];  

    )DOT", fontname, edge_color, penwidth);
    // clang-format on

    print_values(my_graph, dot);
    print_ops(my_graph, dot);
    dot += "\n\t}";

    return dot;
}

void render_graphviz(const std::string &dot, const std::filesystem::path &file_path) {
    std::ofstream file(file_path);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + file_path.string());
    }

    file << dot;
}

namespace {

void print_values(const ONNX_Graph &my_graph, std::string &dot) {
    std::unordered_set<std::string> inserted_value;

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
}

void print_ops(const ONNX_Graph &my_graph, std::string &dot) {
    // operator op_id_map[op_type] create pair with default key = 0
    std::unordered_map<std::string, int> op_id_map;

    for (const auto &op : my_graph.ops_) {
        int op_type_id = op_id_map[op->op_type_]++;
        std::string op_name = op->op_type_ + "_" + std::to_string(op_type_id);

        std::string input_str = op_input_to_string(op->inputs_, op_name);
        printOp(dot, op_name);
        printEdges(op.get(), dot, op_name, input_str);
    }
}

// clang-format off
void printOp(std::string &dot, std::string_view op_name) {
    dot += std::format("\"{}\" [shape=ellipse, style=solid];\n", op_name);
}

void printTensor(const Value* val_ptr, std::string& dot, const TensorStyle& style) {
    dot += std::format(R"DOT(
        "{0}" [shape = none, label=<
            <TABLE BORDER = "1" CELLBORDER = "1" CELLSPACING = "0" CELLPADDING = "4" BGCOLOR = "{1}" COLOR = "{2}">
                <TR><TD HEIGHT="40">{0}</TD></TR>
                <TR><TD HEIGHT="20">{3}</TD></TR>
            </TABLE>
        >];
    )DOT", val_ptr->name_, style.bg_color_, style.color_, shape_to_string(val_ptr->shape_));
}

void printEdges(const Op* op_ptr, std::string& dot, std::string_view op_name, const std::string& input_str) {
    dot += "\n\t\t// inputs of op:\n";
    dot += input_str;
    dot += "\t\t// outputs of op:\n";
    dot += std::format("\t\t\"{}\" -> \"{}\";\n", op_name, op_ptr->outputs_[0]->name_);
}

std::string op_input_to_string(const std::vector<Value *>& input, std::string_view op_name) {
    std::string op_input;
    for (const auto *input_ptr : input) {
        op_input += std::format("\t\t\"{}\" -> \"{}\";\n", input_ptr->name_, op_name);
    }
    
    op_input += "\n";
    return op_input;
}
//clang-format on

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