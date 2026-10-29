#include "format/csr.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

CsrMatrix makeCanonicalCsr(std::uint32_t rows, std::uint32_t cols, std::uint32_t nnz,
    const std::uint32_t* rowOffsets, const std::uint32_t* sourceIds, const float* values) {
    if (!rows || !cols || rowOffsets[0] || rowOffsets[rows] != nnz)
        throw std::invalid_argument("invalid CSR dimensions or offsets");
    for (std::uint32_t row = 0; row < rows; ++row)
        if (rowOffsets[row] > rowOffsets[row+1] || rowOffsets[row+1] > nnz)
            throw std::invalid_argument("invalid CSR offsets");
    CsrMatrix out;
    for (std::uint32_t row = 0; row < rows; ++row) {
        std::vector<std::pair<std::uint32_t, float>> entries;
        for (auto e = rowOffsets[row]; e < rowOffsets[row+1]; ++e) {
            if (sourceIds[e] >= cols || !std::isfinite(values[e]))
                throw std::invalid_argument("invalid CSR column or nonfinite value");
            entries.emplace_back(sourceIds[e], values[e]);
        }
        std::stable_sort(entries.begin(), entries.end(), [](auto a, auto b) { return a.first < b.first; });
        for (std::size_t i = 0; i < entries.size();) {
            auto column = entries[i].first;
            float sum = 0;
            do { sum += entries[i++].second; } while (i < entries.size() && entries[i].first == column);
            if (!std::isfinite(sum)) throw std::invalid_argument("nonfinite coalesced weight");
            if (sum != 0) { out.sourceIds.push_back(column); out.values.push_back(sum); }
        }
        out.rowOffsets.push_back(out.sourceIds.size());
    }
    return out;
}
