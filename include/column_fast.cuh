#pragma once
#include "column_schedule.cuh"

// Fast Column kernels (task-book batch 2): each CTA owns one Column Window
// (8 input columns), loads X once per feature tile and reuses it across the
// CW's tiles; output rows come DIRECTLY from ColumnBaseFormat::tileRows
// (no SW-local slot mapping). The diagnosis (01_diagnosis.md) showed the
// products gap is ~98% output-atomic-RMW; these variants measure how much
// instruction/writeback restructuring can recover, and how much the wide
// (float2/float4) REDG changes throughput. Batch 3 (output restructuring)
// is the decisive path for the 1.5x target; these kernels are the honest
// ablation baseline requested by the task book.

// Non-owning GPU view of the base (SW-independent) column format.
struct DeviceColumnFast {
    std::uint32_t rows = 0, cols = 0;
    const std::uint32_t* colWindowOffset = nullptr; // per-CW tile ranges
    const std::uint32_t* tileRows = nullptr;        // 8 real rows per tile
    const std::uint32_t* tileValueOffset = nullptr;
    const std::uint64_t* tileLocalBit = nullptr;
    const float* values = nullptr;
};

enum FastVariant : int {
    FAST_V1 = 0,   // direct tileRows mapping, scalar atomics, single tile in flight
    FAST_V2A = 1,  // V1 + hoisted address computation (precomputed row*F in smem)
    FAST_V2B = 2,  // V1 + warp reshuffle to emit float2 wide atomics
    FAST_V2C = 3,  // V1 + warp reshuffle to emit float4 wide atomics
    FAST_V3 = 4,   // V1 + tile pipeline (cp.async), stages configurable
    FAST_V4 = 5,   // V1 + internal feature tile FTILE (32/64/128/256)
    FAST_V3W = 6,  // V3 + float2 wide atomics (pipeline x wide combo)
};

struct ColumnFastInfo {
    int registers = 0, sharedBytes = 0, localBytes = 0, activeBlocksPerSM = 0;
};

// features: F of the problem (128/256). ftile: internal feature tile for V4
// (128 otherwise). stages: pipeline depth for V3 (2 or 4, else 1).
cudaError_t launchColumnFast(const DeviceColumnFast& matrix, const float* x, float* y,
    std::uint32_t features, FastVariant variant, std::uint32_t ftile = 128,
    std::uint32_t stages = 1, cudaStream_t stream = nullptr, bool clearOutput = true);
cudaError_t columnFastAttributes(FastVariant variant, std::uint32_t ftile,
    std::uint32_t stages, ColumnFastInfo* info);
