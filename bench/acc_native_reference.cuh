#pragma once
#include "acc_reference.cuh"
cudaError_t launchAccRowSpMMBarrier(const std::uint32_t* window, const std::uint32_t* offset,
    const std::uint32_t* mapping, const std::uint64_t* bits, const float* values,
    const float* x, float* y, std::uint32_t paddedRows, std::uint32_t cols,
    std::uint32_t features, cudaStream_t stream);
