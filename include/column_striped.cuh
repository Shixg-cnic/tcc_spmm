#pragma once
#include "column_fast.cuh"
#include <vector>
#include <cstdint>

// Experimental FP32 column layout grouped by output-row stripes.
// Graph vertex indices are not reordered. Each stripe is processed in launch order.
// A stripe-local (CW,row) is represented in exactly one 8x8 tile.
struct ColumnStriped {
    std::uint32_t rows = 0, cols = 0, stripeRows = 0;
    std::vector<std::uint32_t> stripeCwOffset; // [stripe][cw+1]; flattened (stripes*(cws+1))
    std::vector<std::uint32_t> tileRows;       // absolute row IDs, 8 per tile
    std::vector<std::uint32_t> valueOffset;    // tileCount+1
    std::vector<std::uint64_t> bitmap;         // 8x8 tile, row-major
    std::vector<float> values;                 // packed in bitmap bit order
    std::uint32_t stripes() const { return (rows + stripeRows - 1) / stripeRows; }
    std::uint32_t cws() const { return (cols + 7) / 8; }
};

ColumnStriped makeColumnStriped(const ColumnBaseFormat& src, std::uint32_t stripeRows);

struct DeviceColumnStriped {
    std::uint32_t rows, cols, stripeRows, stripes, cws;
    const std::uint32_t* stripeCwOffset;
    const std::uint32_t* tileRows;
    const std::uint32_t* valueOffset;
    const std::uint64_t* bitmap;
    const float* values;
};

// One launch per stripe on the same stream to keep the active Y region bounded.
// Data construction/upload is outside the timed compute section.
// Scalar output version intended as a correctness reference; WIDE4 uses
// warp shuffles + atomicAdd(float4*) and requires F in {128,256}.
cudaError_t launchColumnStriped(const DeviceColumnStriped& m, const float* x,
    float* y, std::uint32_t features, bool wide4, cudaStream_t stream = nullptr,
    bool clearOutput = true);
