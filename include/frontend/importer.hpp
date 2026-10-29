#pragma once
#include "frontend/graph.hpp"
#include <string>

namespace Labs {

ONNX_Graph importer(const std::string &path);

} // namespace Labs