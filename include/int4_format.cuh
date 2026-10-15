#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// INT4 X storage for the fused kernel: per-node scale + packed nibbles.
//
// Per source row i of X (K rows): s_i = max_j |X[i][j]| / 7, q = clip(round(x/s), -7, 7);
// rows with all-zero X get s_i = 1 and q = 0. Two 4-bit q values are packed per
// byte: feature j even -> low nibble, odd -> high nibble; an odd feature count
// pads the last byte with a zero high nibble. scale is a K-float array.
struct DeviceInt4X {
    const std::uint8_t* q;
    const float* scale;
};

// Host entry points (GPU work on `stream`).
cudaError_t quantizeXInt4(const float* x, std::uint8_t* xq, float* scale,
    std::uint32_t nodes, std::uint32_t features, cudaStream_t stream = nullptr);
// Independent dequantization baseline: writes the FP32 X-hat the FP32 kernels consume.
cudaError_t dequantizeXInt4(const std::uint8_t* xq, const float* scale, float* xhat,
    std::uint32_t nodes, std::uint32_t features, cudaStream_t stream = nullptr);

// Device helpers shared by the quantize kernel and the fused loaders.
// Packing layout: node-major, byte = node*ceil(F/2) + j/2; feature j even ->
// low nibble, odd -> high nibble. nibbleAt takes the BYTE index (from
// packedByteIndex) plus the feature to select the nibble.
__device__ __forceinline__ std::size_t packedByteIndex(std::size_t node,
    std::uint32_t feature, std::uint32_t features) {
    return node * ((std::size_t(features) + 1) / 2) + feature / 2;
}
__device__ __forceinline__ float int4ToFloat(std::uint8_t nibble, float s) {
    int q = (nibble & 8) ? int(nibble) - 16 : int(nibble);
    return float(q) * s;
}
__device__ __forceinline__ std::uint8_t nibbleAt(const std::uint8_t* q,
    std::size_t byteIndex, std::uint32_t feature) {
    return (q[byteIndex] >> ((feature & 1) ? 4 : 0)) & 0xF;
}
