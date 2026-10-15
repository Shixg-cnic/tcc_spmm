#include "row_format.cuh"
#include <algorithm>
#include <unordered_map>

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
