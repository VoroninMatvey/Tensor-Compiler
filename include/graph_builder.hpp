#pragma once
#include "graph.hpp"
#include <filesystem>
#include <string>

namespace Labs {

std::string build_dot(const ONNX_Graph &my_graph, const std::string &fontname,
                      const std::string &edge_color, const double penwidth);
void render_graphviz(const std::string &dot, const std::filesystem::path &file_path);

} // namespace Labs