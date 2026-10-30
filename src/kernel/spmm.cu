#include "spmm.cuh"
#include "cuda_support.cuh"
#include "spmm_kernel.cuh"

namespace {
template <unsigned Features>
void launch(Sw32DeviceView matrix, const std::int8_t* packedX, const float* scales, float* output,
            unsigned rows, unsigned cols, SpmmWorkspace workspace, cudaStream_t stream) {
    // One CTA per panel task and 128-feature slice, with four warps per CTA.
    const dim3 grid(matrix.taskCount, Features / 128);
    spmmInt8Kernel<Features><<<grid, 128, 0, stream>>>(
        matrix, packedX, scales, output, rows, cols, matrix.tasks, workspace.partials,
        matrix.reductions, matrix.partialIds, workspace.counters);
}
} // namespace

void launchSpmmInt8(Sw32DeviceView matrix, const std::int8_t* packedX, const float* scales,
                    float* output, std::uint32_t rows, std::uint32_t cols, std::uint32_t features,
                    SpmmWorkspace workspace, cudaStream_t stream) {
    switch (features) {
    case 128:
        launch<128>(matrix, packedX, scales, output, rows, cols, workspace, stream);
        break;
    case 256:
        launch<256>(matrix, packedX, scales, output, rows, cols, workspace, stream);
        break;
    default:
        throw std::invalid_argument("feature dimension must be 128 or 256");
    }
    checkCuda(cudaGetLastError());
}
