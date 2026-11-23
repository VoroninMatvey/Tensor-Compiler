#include "frontend/importer.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {
template <typename T> void read_tensor(Labs::Value &input, std::ofstream &input_file);
std::size_t element_count(const std::vector<std::int64_t> &shape);

template <typename T> void read_tensor(Labs::Value &input, std::ofstream &input_file) {
    std::size_t count = element_count(input.shape_);

    if (count > input.raw_data_.max_size() / sizeof(T))
        throw std::runtime_error("Tensor is too large : " + input.name_);

    std::size_t byte_count = count * sizeof(T);
    input.raw_data_.resize(byte_count);

    // Using > redirection doesn't write to the file—the output stays in the terminal
    std::cerr << input.name_ << " (" << count << " values):\n";

    for (std::size_t i = 0, elem_size = sizeof(T); i < count; ++i) {
        T value;
        if (!(std::cin >> value))
            throw std::invalid_argument("Invalid input for tensor " + input.name_);

        std::byte *dest = input.raw_data_.data() + elem_size * i;
        std::memcpy(dest, &value, elem_size);
    }

    if (input_file.is_open() && count > 0) {
        const char *data = reinterpret_cast<const char *>(input.raw_data_.data());
        input_file.write(data, static_cast<std::streamsize>(byte_count));
    }
}

std::size_t element_count(const std::vector<std::int64_t> &shape) {
    auto multiply = [](std::size_t acc, int64_t elem) {
        if (elem == -1)
            throw std::runtime_error("Dynamic size of shape\n");

        return acc * static_cast<std::size_t>(elem);
    };

    return std::accumulate(shape.begin(), shape.end(), std::size_t(1), multiply);
}
} // namespace

int main(int argc, char **argv) {
    if (argc != 3 && argc != 4) {
        std::cerr << "Usage: runner <path_to_model.so> <path_to_model.onnx> [save-inputs.bin]\n";
        return 1;
    }

    try {
        void *handle = dlopen(argv[1], RTLD_NOW);
        std::unique_ptr<void, int (*)(void *)> library(handle, dlclose);

        if (!library)
            throw std::runtime_error(dlerror());

        dlerror();
        void *symbol = dlsym(library.get(), "forward_packed");

        if (const char *error = dlerror())
            throw std::runtime_error(error);

        using Forward = void (*)(void **);
        Forward forward = reinterpret_cast<Forward>(symbol);

        Labs::ONNX_Graph graph = Labs::importer(argv[2]);
        if (graph.outputs.empty())
            throw std::invalid_argument("Model has no outputs");

        std::ofstream input_file;
        if (argc == 4) {
            input_file.exceptions(std::ios::failbit | std::ios::badbit);
            input_file.open(argv[3], std::ios::binary);
        }
    } catch (std::exception &excp) {
        std::cerr << excp.what() << "\n";
        return 1;
    }

    return 0;
}