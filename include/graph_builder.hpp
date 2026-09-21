#pragma once
#include "graph.hpp"
#include <string>

namespace Labs {

std::string build_dot(const ONNX_Graph &my_graph, const std::string &fontname = "Arial",
                      const std::string &edge_color = "#555555", const double penwidth = 1.5);
void render_graphviz(const std::string &dot, const std::string &path);

} // namespace Labs