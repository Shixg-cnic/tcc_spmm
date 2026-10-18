#pragma once
#include "row_format.cuh"

RowBitTCF convertAccRow(std::uint32_t rows, std::uint32_t cols, const CanonicalCSR& csr);
cudaError_t launchAccRowSpMM(const std::uint32_t* window, const std::uint32_t* offset,
    const std::uint32_t* mapping, const std::uint64_t* bits, const float* values,
    const float* x, float* y, std::uint32_t paddedRows, std::uint32_t cols,
    std::uint32_t features, cudaStream_t stream);

cudaError_t accRowAttributes(std::uint32_t features,int* registers,int* sharedBytes,int* localBytes,int* blocksPerSM);
