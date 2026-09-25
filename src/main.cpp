#include "graph.hpp"
#include "graph_builder.hpp"
#include "importer.hpp"
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

fs::path prepare_dot_path(const fs::path &onnx_path) {
    fs::path dot_dir = fs::path(PROJECT_ROOT) / "data" / "dot_models";
    fs::create_directories(dot_dir);
    fs::path file = dot_dir / (onnx_path.stem().string() + ".dot");
    return file;
}

void draw_graph(const Labs::ONNX_Graph &my_graph, const fs::path &file) {
    const std::string graph_dot = Labs::build_dot(my_graph, "Arial", "#555555", 1.5);
    Labs::render_graphviz(graph_dot, file);
}

bool has_draw_flag(int argc, char *argv[]) {
    bool draw = false;

    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--draw-dot") == 0)
            draw = true;
    }
    return draw;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <something.onnx> [--draw-dot]\n";
        return 1;
    }

    try {
        Labs::ONNX_Graph my_graph = Labs::importer(argv[1]);

        if (has_draw_flag(argc, argv)) {
            draw_graph(my_graph, prepare_dot_path(argv[1]));
        }
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}