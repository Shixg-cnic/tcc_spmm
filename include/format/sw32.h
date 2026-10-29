#pragma once

#include "format/csr.h"
#include <cstdint>
#include <vector>

// Physical bucket order: four exclusive RWs, followed by shared source panels.
// Slot zero is empty in the bucketed format.
inline constexpr unsigned sw32BucketOrder[16] = {
    0, 1, 2, 4, 8, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15
};

inline constexpr std::uint32_t sw32TaskPanelLimit = 384;
struct Sw32Task {
    std::uint32_t sw, begin, end, partial, mask, chunks, reduceFirst;
};
struct Sw32ReduceRange {
    std::uint32_t begin, end;
};

// CPU storage for 32-row superwindows; each RW is an 8-row group.
// Rows follow the input matrix order; X/scales use the input source ID.
struct Sw32HostFormat {
    std::uint32_t rows = 0, cols = 0;
    std::vector<std::uint32_t> windowOffsets{0}, sourceIds, rowGroupMasks;
    // Per source: low 32 bits = destination-row mask; high 32 bits = value base.
    // Values for a source appear in increasing local destination-row order.
    std::vector<std::uint64_t> sourceMetadata;
    std::vector<float> values;
    // When RW buckets are requested: 17 absolute panel offsets per SW.
    // Slot s (mask sw32BucketOrder[s]) occupies [offset[s], offset[s+1]).
    // Slots 1..4 are exclusive RWs; slots 5..15 contain shared panels.
    std::vector<std::uint32_t> bucketOffsets;
    // Natural SW order. Heavy SWs have bounded, contiguous panel tasks.
    // Split tasks store only their participating 8-row groups in partials.
    std::vector<Sw32Task> tasks;
    std::vector<Sw32ReduceRange> reductions;
    std::vector<std::uint32_t> partialIds;
    std::uint32_t partialCount = 0;
};

// CPU conversion only. Rows and source IDs follow the input matrix.
// Keep bucketByRwMask=true for launchSpmmInt8().
Sw32HostFormat convertCsrToSw32(std::uint32_t rows, std::uint32_t cols,
    const CsrMatrix& csr, bool bucketByRwMask = true);

// Non-owning GPU pointers to the corresponding arrays above; upload is explicit.
struct Sw32DeviceView {
    const std::uint32_t *windowOffsets, *sourceIds, *rowGroupMasks;
    const std::uint64_t* sourceMetadata;
    const float* values;
    const std::uint32_t* bucketOffsets = nullptr;
    const Sw32Task* tasks = nullptr;
    const Sw32ReduceRange* reductions = nullptr;
    const std::uint32_t* partialIds = nullptr;
    std::uint32_t taskCount = 0;
};

