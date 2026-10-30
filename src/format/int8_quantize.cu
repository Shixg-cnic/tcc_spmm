#include "format/int8_features.cuh"
#include "cuda_support.cuh"
namespace {
__global__ void quantize(const float* x, std::int8_t* q, float* scale, std::uint32_t f) {
    __shared__ float maxima[128];
    std::uint32_t row = blockIdx.x, t = threadIdx.x;
    float m = 0;
    for (std::uint32_t j = t; j < f; j += blockDim.x)
        m = fmaxf(m, fabsf(x[std::size_t(row) * f + j]));
    maxima[t] = m;
    __syncthreads();
    for (std::uint32_t step = 64; step; step >>= 1) {
        if (t < step)
            maxima[t] = fmaxf(maxima[t], maxima[t + step]);
        __syncthreads();
    }
    float s = maxima[0] == 0 ? 1 : maxima[0] / 127.f;
    if (t == 0)
        scale[row] = s;
    for (std::uint32_t j = t; j < f; j += blockDim.x) {
        auto i = std::size_t(row) * f + j;
        int v = lrintf(x[i] / s);
        q[i] = std::int8_t(v < -127 ? -127 : (v > 127 ? 127 : v));
    }
}
} // namespace
void quantizeFeaturesInt8(const float* x, std::int8_t* q, float* scale, std::uint32_t n,
                          std::uint32_t f, cudaStream_t stream) {
    quantize<<<n, 128, 0, stream>>>(x, q, scale, f);
    checkCuda(cudaGetLastError());
}
