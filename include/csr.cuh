#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

// Canonical CSR shared by the Row and Column pipelines.
//
// Row pointers / source columns / FP32 weights with duplicate coordinates
// already summed, exact-zero sums removed, and every row sorted ascending by
// source column. Both the Row-BitTCF and Column-SWTCF converters consume this
// structure; it is deliberately independent of any Tensor-Core layout so that
// Column does not depend on Row format headers.
struct CanonicalCSR {
    std::vector<std::uint32_t> ptr{0}, src;
    std::vector<float> values;
};

// Validate raw CSR arrays and build a CanonicalCSR: stable-sort each row by
// source column, coalesce duplicate coordinates, drop zero-sum entries and
// reject non-finite weights or invalid offsets.
CanonicalCSR canonicalCSR(std::uint32_t rows, std::uint32_t cols, std::uint32_t nnz,
    const std::uint32_t* ptr, const std::uint32_t* src, const float* values);
