#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// X is row-major [source, feature]. Quantize each source independently:
// scale=max(abs(X[source,:]))/127; all-zero rows use scale=1; q is in [-127,127].
cudaError_t quantizeFeaturesInt8(const float* x, std::int8_t* q, float* scale,
    std::uint32_t sources, std::uint32_t features, cudaStream_t stream = nullptr);
// Standalone reconstruction for validation. SpMM fuses this inside registers.
cudaError_t dequantizeFeaturesInt8(const std::int8_t* q, const float* scale, float* x,
    std::uint32_t sources, std::uint32_t features, cudaStream_t stream = nullptr);
// Physical byte layout only: each 16-feature tile is [q0,q8,q1,q9,...,q7,q15].
// Output has sources*features bytes and must not alias q. No values are changed.
cudaError_t packFeaturesInt8x16(const std::int8_t* q, std::int8_t* packed,
    std::uint32_t sources, std::uint32_t features, cudaStream_t stream = nullptr);
