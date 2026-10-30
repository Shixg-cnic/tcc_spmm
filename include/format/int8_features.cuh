#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// Callers provide valid, non-aliasing buffers and nonzero dimensions.
// CUDA launch failures throw std::runtime_error; execution remains asynchronous.
// X is row-major [source, feature]. Quantize each source independently:
// scale=max(abs(X[source,:]))/127; all-zero rows use scale=1; q is in [-127,127].
void quantizeFeaturesInt8(const float* x, std::int8_t* q, float* scale, std::uint32_t sources,
                          std::uint32_t features, cudaStream_t stream = nullptr);
// Physical byte layout only: each 16-feature tile is [q0,q8,q1,q9,...,q7,q15].
// Features must be a multiple of 16. Output has sources*features bytes,
// must not alias q, and must be aligned for unsigned-short access. Values are unchanged.
void packFeaturesInt8x16(const std::int8_t* q, std::int8_t* packed, std::uint32_t sources,
                         std::uint32_t features, cudaStream_t stream = nullptr);
