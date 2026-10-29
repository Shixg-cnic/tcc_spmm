#pragma once
#include "format/sw32.h"
#include <cuda_runtime.h>

// Per-invocation storage, reusable by serial launches on the same stream.
// partials: Sw32HostFormat::partialCount * 8 * features floats.
// counters: ceil(rows/32) * (features/128) uint32 values, initially zero.
// Both may be null when partialCount is zero. Use separate workspaces for
// concurrent launches; keep all buffers alive until the stream completes.
struct SpmmWorkspace {
    float* partials = nullptr;
    std::uint32_t* counters = nullptr;
};

// F128/F256 fused INT8 dequantization + SpMM, including partial reduction.
// q is packed16, scale is per source, y is FP32. The default RW-bucketed
// SW32 format includes the bounded task plan; upload all its arrays.
// Every output is overwritten. Counters return to zero after each launch.
cudaError_t launchSpmmInt8(Sw32DeviceView format,
    const std::int8_t* qPacked16, const float* scale, float* y,
    std::uint32_t rows, std::uint32_t cols, std::uint32_t features,
    SpmmWorkspace workspace, cudaStream_t stream = nullptr);
