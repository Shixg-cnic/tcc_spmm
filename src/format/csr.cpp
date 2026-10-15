#include "csr.cuh"
#include <algorithm>
#include <cmath>
#include <stdexcept>

CanonicalCSR canonicalCSR(std::uint32_t rows, std::uint32_t cols, std::uint32_t nnz,
    const std::uint32_t* ptr, const std::uint32_t* src, const float* values) {
    if (!rows || !cols || ptr[0] || ptr[rows] != nnz)
        throw std::invalid_argument("invalid CSR dimensions or offsets");
    for (std::uint32_t row = 0; row < rows; ++row)
        if (ptr[row] > ptr[row+1] || ptr[row+1] > nnz)
            throw std::invalid_argument("invalid CSR offsets");
    CanonicalCSR out;
    for (std::uint32_t row = 0; row < rows; ++row) {
        std::vector<std::pair<std::uint32_t, float>> entries;
        for (auto e = ptr[row]; e < ptr[row+1]; ++e) {
            if (src[e] >= cols || !std::isfinite(values[e]))
                throw std::invalid_argument("invalid CSR column or nonfinite value");
            entries.emplace_back(src[e], values[e]);
        }
        std::stable_sort(entries.begin(), entries.end(), [](auto a, auto b) { return a.first < b.first; });
        for (std::size_t i = 0; i < entries.size();) {
            auto column = entries[i].first;
            float sum = 0;
            do { sum += entries[i++].second; } while (i < entries.size() && entries[i].first == column);
            if (!std::isfinite(sum)) throw std::invalid_argument("nonfinite coalesced weight");
            if (sum != 0) { out.src.push_back(column); out.values.push_back(sum); }
        }
        out.ptr.push_back(out.src.size());
    }
    return out;
}
