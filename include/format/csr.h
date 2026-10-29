#pragma once
#include <cstdint>
#include <string>
#include <vector>

// Canonical CSR input: sorted source columns and FP32 weights.
struct CsrMatrix {
    std::vector<std::uint32_t> rowOffsets{0}, sourceIds;
    std::vector<float> values;
};

// Read a Matrix Market coordinate matrix into canonical CPU CSR.
struct MtxMatrix {
    std::uint32_t rows, cols;
    CsrMatrix csr;
};

MtxMatrix loadMatrixMarket(const std::string& path);

// Validate raw CSR arrays and build a CsrMatrix: stable-sort each row by
// source column, coalesce duplicate coordinates, drop zero-sum entries and
// reject non-finite weights or invalid offsets.
CsrMatrix makeCanonicalCsr(std::uint32_t rows, std::uint32_t cols, std::uint32_t nnz,
    const std::uint32_t* rowOffsets, const std::uint32_t* sourceIds, const float* values);

