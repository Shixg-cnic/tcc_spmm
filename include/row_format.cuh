#pragma once
#include "csr.cuh"

struct RowBitTCF {
    std::uint32_t rows, cols;
    std::vector<std::uint32_t> rowWindowOffset{0}, tcOffset{0}, sparseA2B;
    std::vector<std::uint64_t> tcLocalBit;
    std::vector<float> values;
};

RowBitTCF convertRow(std::uint32_t rows, std::uint32_t cols, const CanonicalCSR& csr);

cudaError_t launchRowSpMM(const std::uint32_t* window, const std::uint32_t* offset,
    const std::uint32_t* mapping, const std::uint64_t* bits, const float* values,
    const float* x, float* y, std::uint32_t rows, std::uint32_t cols,
    std::uint32_t features, cudaStream_t stream = nullptr);
cudaError_t rowKernelAttributes(int* attributes);
