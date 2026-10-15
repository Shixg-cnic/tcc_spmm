#pragma once
#include "column_format.cuh"   // COLUMN_FEATURE_TILE / COLUMN_CACHE_SIZE / COLUMN_CACHE_STRIDE

// Dynamic SuperWindow scheduling for the Column pipeline.
//
// The base format is SW-independent: every tile stores its real target rows,
// so a schedule can be rebuilt (or a CW split across tasks) without re-sorting
// the CSR or rebuilding the tile stream. The schedule groups CWs into variable
// length SuperWindows (SWTasks); the adaptive format materialises the SW-local
// row slots / cache directory and the GPU task table that the kernel consumes.

struct ColumnBaseFormat {
    std::uint32_t rows = 0, cols = 0;
    std::vector<std::uint32_t> colWindowOffset{0};   // per-CW tile ranges
    std::vector<std::uint64_t> tileLocalBit;
    std::vector<std::uint32_t> tileValueOffset{0};
    std::vector<float> values;
    std::vector<std::uint32_t> tileRows;             // 8 real rows per tile, UINT32_MAX = padding
};

// One CTA task: a variable-length SW, possibly a chunk of a split heavy CW.
struct SWTask {
    std::uint32_t cwBegin = 0, cwEnd = 0;            // [cwBegin, cwEnd)
    std::uint32_t tileBegin = 0, tileEnd = 0;        // absolute tile range
    std::uint32_t rowBase = 0;                       // offset into swRows for this SW
    std::uint32_t swId = 0;                          // SW index (== task index for grouping tasks)
    std::uint32_t flags = 0;                         // 1 = split chunk of a heavy CW
};
enum SWTaskFlags : std::uint32_t { SWT_SPLIT_CHUNK = 1 };

struct ColumnSchedule {
    std::vector<SWTask> tasks;
    std::vector<std::uint32_t> swRowOffset{0};       // per-SW unique-row mapping
    std::vector<std::uint32_t> swRows;               // real row ids, sorted per SW
    std::vector<std::uint32_t> swRowCacheSlot;       // sw-local row slot -> cache slot or UINT32_MAX
    std::vector<std::uint32_t> swCachedRowSlot;      // 32 cache slots -> sw-local row slot or UINT32_MAX
    std::vector<std::uint32_t> swCachedCount;        // valid cache rows per SW
    std::uint32_t splitChunks = 0;                   // how many tasks came from heavy-CW splits
};

struct ScheduleConfig {
    std::uint32_t maxTilesPerTask = 512;             // tile workload cap per task
    std::uint32_t maxCwsPerSw = 64;                  // CW count cap per SW
    std::uint32_t splitTileThreshold = 384;          // CWs above this get split into chunks
    std::uint32_t splitChunkTiles = 256;             // chunk size for heavy-CW splits
    double minReuseSavingFraction = 0.002;           // merge only if reuse saves >= this fraction
};

// The GPU-consumed adaptive format: base tile stream + SW-local slots + tasks.
struct ColumnAdaptiveFormat {
    std::uint32_t rows = 0, cols = 0;
    std::vector<std::uint32_t> colWindowOffset{0};   // per-CW tile ranges (absolute)
    std::vector<std::uint32_t> tileRowSlot;          // 8 SW-local slots per tile, UINT32_MAX = padding
    std::vector<std::uint32_t> tileValueOffset{0};
    std::vector<std::uint64_t> tileLocalBit;
    std::vector<float> values;
    std::vector<std::uint32_t> swRowOffset{0}, swRows;
    std::vector<std::uint32_t> swRowCacheSlot, swCachedRowSlot, swCachedCount;
    std::vector<SWTask> tasks;
};

// Non-owning GPU view; caller manages device memory.
struct DeviceColumnAdaptive {
    std::uint32_t rows, cols, taskCount;
    const std::uint32_t* colWindowOffset;
    const std::uint32_t* tileRowSlot;
    const std::uint32_t* tileValueOffset;
    const std::uint64_t* tileLocalBit;
    const float* values;
    const std::uint32_t* swRowOffset;
    const std::uint32_t* swRows;
    const std::uint32_t* swRowCacheSlot;
    const std::uint32_t* swCachedRowSlot;
    const std::uint32_t* swCachedCount;
    const SWTask* tasks;
};

ColumnBaseFormat buildColumnBase(std::uint32_t rows, std::uint32_t cols, const CanonicalCSR& csr);
ColumnSchedule buildAdaptiveSchedule(const ColumnBaseFormat& base,
    const ScheduleConfig& config = ScheduleConfig{});
ColumnAdaptiveFormat buildAdaptiveColumn(const ColumnBaseFormat& base, const ColumnSchedule& schedule);

// Adaptive kernel launches like the fixed-SW ones: complete includes memset.
// useCache selects the C3-style SW output cache (default true).
cudaError_t launchColumnAdaptiveSpMM(const DeviceColumnAdaptive& matrix, const float* x, float* y,
    std::uint32_t features, bool useCache, cudaStream_t stream = nullptr, bool clearOutput = true);
// Fused INT4 variant (x.q/x.scale packed+scaled, see int4_format.cuh).
cudaError_t launchColumnAdaptiveInt4SpMM(const DeviceColumnAdaptive& matrix, const DeviceInt4X& x, float* y,
    std::uint32_t features, bool useCache, cudaStream_t stream = nullptr, bool clearOutput = true);
