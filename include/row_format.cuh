#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

struct RowBitTCF {
    std::uint32_t rows, cols;
    std::vector<std::uint32_t> rowWindowOffset{0}, tcOffset{0}, sparseA2B;
    std::vector<std::uint64_t> tcLocalBit;
    std::vector<float> values;
};

struct CanonicalCSR {
    std::vector<std::uint32_t> ptr{0}, src;
    std::vector<float> values;
};

CanonicalCSR canonicalCSR(std::uint32_t rows, std::uint32_t cols, std::uint32_t nnz,
    const std::uint32_t* ptr, const std::uint32_t* src, const float* values);
RowBitTCF convertRow(std::uint32_t rows, std::uint32_t cols, const CanonicalCSR& csr);

cudaError_t launchRowSpMM(const std::uint32_t* window, const std::uint32_t* offset,
    const std::uint32_t* mapping, const std::uint64_t* bits, const float* values,
    const float* x, float* y, std::uint32_t rows, std::uint32_t cols,
    std::uint32_t features, cudaStream_t stream = nullptr);
cudaError_t rowKernelAttributes(int* attributes);
