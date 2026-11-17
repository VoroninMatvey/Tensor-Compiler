#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace Labs {
class MemRef {
  public:
    static constexpr std::size_t kSlot = sizeof(void *);
    static_assert(sizeof(void *) == sizeof(std::intptr_t));

    // To pass the mlir_ciface_forward argument to the output
    explicit MemRef(std::size_t rank) : rank_(rank), bytes_((3 + 2 * rank) * kSlot) {}

    // To pass the mlir_ciface_forward argument to the input
    MemRef(const std::vector<int64_t> &shape, std::byte *data) : MemRef(shape.size()) {
        set_ptr(0, data);
        set_ptr(1, data);
        set_int(2, 0);

        std::intptr_t mul = 1;
        std::size_t rank = shape.size();
        for (std::size_t i = 0; i < rank; ++i) {
            std::intptr_t cur = static_cast<intptr_t>(shape[rank - i - 1]);

            set_int(2 + rank - i, cur);
            set_int(2 + 2 * rank - i, mul);
            mul *= cur;
        }
    }

    // To read the output argument of mlir ciface_forward
    static MemRef from_raw(const std::byte *src, std::size_t rank) {
        MemRef out(rank);
        std::memcpy(out.bytes_.data(), src, out.bytes_.size());
        return out;
    }

  private:
    std::size_t rank_;
    std::vector<std::byte> bytes_;

    void set_ptr(std::size_t s, void *p) { std::memcpy(&bytes_[s * kSlot], &p, sizeof(p)); }
    void set_int(std::size_t s, std::intptr_t x) { std::memcpy(&bytes_[s * kSlot], &x, sizeof(x)); }

    void *get_ptr(std::size_t s) const {
        void *p;
        std::memcpy(&p, &bytes_[s * kSlot], sizeof(p));
        return p;
    }

    std::intptr_t get_int(std::size_t s) const {
        std::intptr_t x;
        std::memcpy(&x, &bytes_[s * kSlot], sizeof(x));
        return x;
    }

  public:
    void *raw() { return bytes_.data(); }
    void *allocated() { return get_ptr(0); }
    void *aligned() { return get_ptr(1); }
    std::intptr_t offset() { return get_int(2); }
    std::intptr_t dim_i(std::size_t i) { return get_int(3 + i); }
};
} // namespace Labs