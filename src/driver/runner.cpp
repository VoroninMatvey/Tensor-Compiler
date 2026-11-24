#include "driver/runner.hpp"
#include "frontend/importer.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

using Forward = void (*)(void **);

namespace {
template <typename T> void read_tensor(Labs::Value &input, std::ofstream &input_file);
template <typename T> void print_values(Labs::MemRef &memref, std::size_t count);
std::size_t element_count(const std::vector<std::int64_t> &shape);

std::vector<Labs::MemRef> read_inputs(Labs::ONNX_Graph &graph, std::ofstream &input_file) {
    std::vector<Labs::MemRef> memrefs;
    memrefs.reserve(graph.inputs.size());

    for (auto *input : graph.inputs) {
        switch (input->type_) {
        case Labs::DataType::Float32:
            read_tensor<float>(*input, input_file);
            break;
        case Labs::DataType::Int64:
            read_tensor<int64_t>(*input, input_file);
            break;
        default:
            throw std::runtime_error("Unsupported type of input:" + input->name_);
        }

        memrefs.emplace_back(input->shape_, input->raw_data_.data());
    }

    if (input_file.is_open())
        input_file.close();

    return memrefs;
}

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

std::size_t descriptor_slots(std::size_t rank) { return 3 + 2 * rank; }

std::vector<std::byte> allocate_outputs(const Labs::ONNX_Graph &graph) {
    std::size_t total_slots = 0;

    for (const auto *output : graph.outputs)
        total_slots += descriptor_slots(output->shape_.size());

    std::size_t total_bytes = total_slots * Labs::MemRef::kSlot;
    return std::vector<std::byte>(total_bytes);
}

void call_forward(Forward forward, std::vector<Labs::MemRef> &inputs,
                  std::vector<std::byte> &output_descriptors) {
    std::vector<void *> args;
    args.reserve(inputs.size() + 1);

    args.push_back(output_descriptors.data());
    for (auto &input_memref : inputs)
        args.push_back(input_memref.raw());

    forward(args.data());
}

std::vector<Labs::MemRef> read_outputs(const Labs::ONNX_Graph &graph,
                                       const std::vector<std::byte> &output_descriptors) {
    std::vector<Labs::MemRef> memrefs;
    memrefs.reserve(graph.outputs.size());
    const std::byte *cur_pointer = output_descriptors.data();

    for (const auto *output : graph.outputs) {
        std::size_t cur_rank = output->shape_.size();
        memrefs.push_back(Labs::MemRef::from_raw(cur_pointer, cur_rank));
        cur_pointer += descriptor_slots(cur_rank) * Labs::MemRef::kSlot;
    }

    return memrefs;
}

void print_outputs(const Labs::ONNX_Graph &graph, std::vector<Labs::MemRef> &outputs) {
    std::cout << std::setprecision(std::numeric_limits<float>::max_digits10);

    for (std::size_t i = 0, size = outputs.size(); i < size; ++i) {
        const Labs::Value &output = *graph.outputs[i];
        Labs::MemRef &memref = outputs[i];

        for (std::size_t d = 0; d < output.shape_.size(); ++d) {
            if (memref.dim_i(d) != output.shape_[d])
                throw std::runtime_error("Unexpected output shape: " + output.name_);
        }

        std::size_t count = element_count(output.shape_);
        std::cout << output.name_ << "\n";

        switch (output.type_) {
        case Labs::DataType::Float32:
            print_values<float>(memref, count);
            break;
        case Labs::DataType::Int64:
            print_values<int64_t>(memref, count);
            break;
        default:
            throw std::runtime_error("Unsupported output type: " + output.name_);
        }
    }
}

template <typename T> void print_values(Labs::MemRef &memref, std::size_t count) {
    if (count == 0) {
        std::cout << "\n";
        return;
    }

    const T *data = static_cast<const T *>(memref.aligned());
    if (!data)
        throw std::runtime_error("Output buffer is null");

    data += memref.offset();

    for (std::size_t i = 0; i < count; ++i) {
        if (i > 0)
            std::cout << " ";
        std::cout << data[i];
    }
    std::cout << "\n";
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
        Forward forward = reinterpret_cast<Forward>(symbol);

        Labs::ONNX_Graph graph = Labs::importer(argv[2]);
        if (graph.outputs.empty())
            throw std::invalid_argument("Model has no outputs");

        std::ofstream input_file;
        if (argc == 4) {
            input_file.exceptions(std::ios::failbit | std::ios::badbit);
            input_file.open(argv[3], std::ios::binary);
        }

        std::vector<Labs::MemRef> input_memrefs = read_inputs(graph, input_file);
        std::vector<std::byte> output_descriptors = allocate_outputs(graph);
        call_forward(forward, input_memrefs, output_descriptors);
        std::vector<Labs::MemRef> outputs = read_outputs(graph, output_descriptors);
        print_outputs(graph, outputs);
    } catch (std::exception &excp) {
        std::cerr << excp.what() << "\n";
        return 1;
    }

    return 0;
}