#include "row_format.cuh"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

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

RowBitTCF convertRow(std::uint32_t rows, std::uint32_t cols, const CanonicalCSR& csr) {
    RowBitTCF out;
    out.rows = rows; out.cols = cols;
    for (std::uint32_t start = 0; start < rows; start += 8) {
        auto end = std::min(start + 8, rows);
        std::vector<std::uint32_t> unique(csr.src.begin()+csr.ptr[start], csr.src.begin()+csr.ptr[end]);
        std::sort(unique.begin(), unique.end());
        unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
        std::unordered_map<std::uint32_t, std::uint32_t> local;
        for (std::uint32_t i = 0; i < unique.size(); ++i) local[unique[i]] = i;
        std::size_t tiles = (unique.size() + 7) / 8;
        unique.resize(tiles * 8, UINT32_MAX);
        out.sparseA2B.insert(out.sparseA2B.end(), unique.begin(), unique.end());
        std::vector<std::uint64_t> bits(tiles, 0);
        std::vector<std::vector<float>> tileValues(tiles);
        for (auto row = start; row < end; ++row) {
            for (auto e = csr.ptr[row]; e < csr.ptr[row+1]; ++e) {
                auto index = local.at(csr.src[e]);
                bits[index / 8] |= std::uint64_t{1} << ((row - start) * 8 + index % 8);
                tileValues[index / 8].push_back(csr.values[e]);
            }
        }
        for (std::size_t tile = 0; tile < tiles; ++tile) {
            out.tcLocalBit.push_back(bits[tile]);
            out.values.insert(out.values.end(), tileValues[tile].begin(), tileValues[tile].end());
            out.tcOffset.push_back(out.values.size());
        }
        out.rowWindowOffset.push_back(out.tcLocalBit.size());
    }
    return out;
}
