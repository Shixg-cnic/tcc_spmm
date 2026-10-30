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
// packedX is packed16, scales are per source, output is FP32. The default RW-bucketed
// SW32 format includes the bounded task plan; upload all its arrays.
// Callers supply valid device views and appropriately sized buffers.
// CUDA launch failures throw std::runtime_error; execution remains asynchronous.
// Every output is overwritten. Counters return to zero after each launch.
void launchSpmmInt8(Sw32DeviceView matrix, const std::int8_t* packedX, const float* scales,
                    float* output, std::uint32_t rows, std::uint32_t cols, std::uint32_t features,
                    SpmmWorkspace workspace, cudaStream_t stream = nullptr);
