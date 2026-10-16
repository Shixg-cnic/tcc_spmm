#pragma once

#include "column_schedule.cuh"

#include <cstdint>
#include <vector>

// Offline CW execution order (experiment/cw-order-only, task-book section 2/3).
//
// The GPU Column kernel is unchanged except for a task-position -> CW-id
// lookup: `w = taskOrder[blockIdx.x]`. Only this permutation array is added;
// no (CW,row) pair copies, no dense CWxRow matrix, no O(W^2) work.
//
// Algorithms (all O(rows+nnz_effective + W log W)):
//   Identity     : order[w]=w (control: cost of the task-mapping mechanism)
//   SeededRandom : fixed-seed shuffle (control: does any reorder help?)
//   HotAnchor    : sort CWs by their highest-frequency output row
//   BandAnchor   : sort CWs by dominant 16384-row band, then anchor row
enum class CWOrderStrategy : std::uint32_t {
    Identity = 0,
    SeededRandom = 1,
    HotAnchor = 2,
    BandAnchor = 3
};

struct CWOrderConfig {
    CWOrderStrategy strategy = CWOrderStrategy::Identity;
    std::uint32_t bandRows = 16384;   // BandAnchor band width in output rows
    std::uint32_t seed = 20261016;    // SeededRandom shuffle seed
};

struct CWOrderResult {
    std::vector<std::uint32_t> order; // strictly all CW ids, no dup/missing
    double buildMs = 0.0;
};

CWOrderResult buildCWOrder(const ColumnBaseFormat& matrix, const CWOrderConfig& config);

// Throws std::runtime_error on size mismatch, out-of-range or duplicated CW.
void validateCWOrder(const std::vector<std::uint32_t>& order, std::uint32_t numWindows);

// ---- offline locality statistics (task-book section 5) ----

struct CWReuseStats {
    std::uint32_t groupSize = 0;
    double reuseRatio = 0;            // sum(visits-unique)/sum(visits) over groups
    double uniqueRowsMean = 0, uniqueRowsP95 = 0, uniqueRowsMax = 0;
    double yBytesMean = 0, yBytesP95 = 0, yBytesMax = 0;  // uniqueRows*F*4
};

struct CWOrderStructureStats {
    std::uint32_t numWindows = 0, totalTiles = 0;
    std::uint64_t totalRowOccurrences = 0;   // effective (CW,row) pairs
    double dominantBandConcentration = 0;    // mean over CWs of maxBand/|Rows| (BandAnchor metric, 0 if no CW)
    CWReuseStats reuse[3];                   // group sizes 32/64/128
};

// Computes the group-reuse statistics for one permutation using the
// lastSeen[rows] trick (linear in total row occurrences). groupSizes = {32,64,128}.
// bandRows is used only for dominantBandConcentration.
CWOrderStructureStats analyzeCWOrder(const ColumnBaseFormat& matrix,
    const std::vector<std::uint32_t>& order, std::uint32_t bandRows,
    const std::uint32_t groupSizes[3]);
