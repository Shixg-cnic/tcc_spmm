#include "../tests/acc_reference.cu"
#include "acc_native_barrier.cuh"
#include "acc_native_reference.cuh"
cudaError_t launchAccRowSpMMBarrier(const std::uint32_t* window, const std::uint32_t* offset,
    const std::uint32_t* mapping, const std::uint64_t* bits, const float* values,
    const float* x, float* y, std::uint32_t paddedRows, std::uint32_t cols,
    std::uint32_t features, cudaStream_t stream) {
    if (paddedRows % 8 || features < 64 || features > 512 || features % 32 ||
        std::uint64_t(cols) * features > UINT32_MAX ||
        std::uint64_t(paddedRows) * features > UINT32_MAX) return cudaErrorInvalidValue;
    tf32_mma_kernel_barrier<<<dim3(paddedRows/8), dim3(32,features/32), 0, stream>>>(
        bits, mapping, values, window, offset, x, y, cols, features);
    return cudaGetLastError();
}

