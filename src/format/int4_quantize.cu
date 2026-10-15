// INT4 quantization / dequantization.
//
// quantizeXInt4: per-node scale s = max|X|/7 (all-zero row -> s=1, q=0),
// q = clip(round(x/s), -7, 7), packed two nibbles per byte (even feature ->
// low nibble, odd -> high; odd features pad the last byte).
// dequantizeXInt4: independent baseline that materialises FP32 X-hat from the
// packed stream for the existing FP32 kernels.
#include "int4_format.cuh"
#include <limits>

namespace {
using U = std::uint32_t;

__global__ void quantizeKernel(const float* x, std::uint8_t* q, float* scale, U features) {
    __shared__ float smax[128];
    const U node = blockIdx.x;
    const U tid = threadIdx.x;
    float m = 0.f;
    for (U j = tid; j < features; j += blockDim.x)
        m = fmaxf(m, fabsf(x[std::size_t(node) * features + j]));
    smax[tid] = m;
    __syncthreads();
    for (U s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smax[tid] = fmaxf(smax[tid], smax[tid + s]);
        __syncthreads();
    }
    const float s = smax[0] == 0.f ? 1.f : smax[0] / 7.f;
    if (tid == 0) scale[node] = s;
    const U pairs = (features + 1) / 2;
    for (U p = tid; p < pairs; p += blockDim.x) {
        const U j0 = 2 * p, j1 = j0 + 1;
        float v0 = x[std::size_t(node) * features + j0] / s;
        int r0 = lrintf(v0);
        r0 = r0 < -7 ? -7 : (r0 > 7 ? 7 : r0);
        std::uint8_t byte = std::uint8_t(r0) & 0xF;
        if (j1 < features) {
            float v1 = x[std::size_t(node) * features + j1] / s;
            int r1 = lrintf(v1);
            r1 = r1 < -7 ? -7 : (r1 > 7 ? 7 : r1);
            byte |= (std::uint8_t(r1) & 0xF) << 4;
        }
        q[std::size_t(node) * pairs + p] = byte;
    }
}

__global__ void dequantKernel(const std::uint8_t* q, const float* scale, float* out,
    U nodes, U features) {
    const U pairs = (features + 1) / 2;
    std::size_t stride = std::size_t(gridDim.x) * blockDim.x;
    for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < std::size_t(nodes) * pairs; i += stride) {
        const U node = U(i / pairs), p = U(i % pairs);
        const U j0 = 2 * p;
        const float s = scale[node];
        out[std::size_t(node) * features + j0] =
            int4ToFloat(nibbleAt(q, packedByteIndex(node, j0, features), j0), s);
        if (j0 + 1 < features)
            out[std::size_t(node) * features + j0 + 1] =
                int4ToFloat(nibbleAt(q, packedByteIndex(node, j0 + 1, features), j0 + 1), s);
    }
}
} // namespace

cudaError_t quantizeXInt4(const float* x, std::uint8_t* xq, float* scale,
    U nodes, U features, cudaStream_t stream) {
    if (!x || !xq || !scale || !nodes || !features) return cudaErrorInvalidValue;
    if (features > std::numeric_limits<U>::max()) return cudaErrorInvalidValue;
    quantizeKernel<<<nodes, 128, 0, stream>>>(x, xq, scale, features);
    return cudaGetLastError();
}

cudaError_t dequantizeXInt4(const std::uint8_t* xq, const float* scale, float* xhat,
    U nodes, U features, cudaStream_t stream) {
    if (!xq || !scale || !xhat || !nodes || !features) return cudaErrorInvalidValue;
    const U pairs = (features + 1) / 2;
    const std::size_t total = std::size_t(nodes) * pairs;
    const U threads = 256;
    const U blocks = U((total + threads - 1) / threads);
    dequantKernel<<<blocks, threads, 0, stream>>>(xq, scale, xhat, nodes, features);
    return cudaGetLastError();
}
