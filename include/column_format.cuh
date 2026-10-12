#pragma once
#include "row_format.cuh"

#ifndef TCC_SW_SIZE
#define TCC_SW_SIZE 16
#endif
#ifndef TCC_CACHE_SIZE
#define TCC_CACHE_SIZE 32
#endif
constexpr std::uint32_t COLUMN_SW_SIZE = TCC_SW_SIZE;
constexpr std::uint32_t COLUMN_CACHE_SIZE = TCC_CACHE_SIZE;
constexpr std::uint32_t COLUMN_FEATURE_TILE = 128;
static_assert(TCC_SW_SIZE > 0 && TCC_CACHE_SIZE > 0);
static_assert(std::uint64_t(COLUMN_CACHE_SIZE)*COLUMN_FEATURE_TILE <= UINT32_MAX);

struct ColumnSWTCF {
    std::uint32_t rows = 0, cols = 0, superWindowSize = COLUMN_SW_SIZE;
    std::vector<std::uint32_t> colWindowOffset{0}, tileRowSlot, tileValueOffset{0};
    std::vector<std::uint64_t> tileLocalBit;
    std::vector<float> values;
    std::vector<std::uint32_t> superWindowRowOffset{0}, superWindowRows;
    // All row slots are local to their SW. UINT32_MAX denotes padding / no cache.
    std::vector<std::uint32_t> superWindowRowCacheSlot, superWindowCachedRowSlot, superWindowCachedCount;
};
ColumnSWTCF convertColumn(std::uint32_t rows, std::uint32_t cols,
    const CanonicalCSR& csr, std::uint32_t superWindowSize = COLUMN_SW_SIZE);

// Non-owning view: every pointer refers to GPU memory; caller manages its lifetime.
struct DeviceColumnSWTCF {
    std::uint32_t rows, cols, superWindowSize;
    const std::uint32_t *colWindowOffset, *tileRowSlot, *tileValueOffset;
    const std::uint64_t* tileLocalBit;
    const float* values;
    const std::uint32_t *superWindowRowOffset, *superWindowRows, *superWindowRowCacheSlot;
    const std::uint32_t *superWindowCachedRowSlot, *superWindowCachedCount;
};
enum class ColumnVersion { C0, C1, C2, C3 };
// By default includes asynchronous Y initialization. Set clearOutput=false only
// when Y was explicitly zeroed on this stream (e.g. isolated kernel timing).
cudaError_t launchColumnSpMM(const DeviceColumnSWTCF& matrix, const float* x, float* y,
    std::uint32_t features, ColumnVersion version, cudaStream_t stream = nullptr,
    bool clearOutput = true);
struct ColumnKernelInfo {
    int registers, sharedBytes, localBytes, activeBlocksPerSM;
};
cudaError_t columnKernelAttributes(ColumnVersion version, ColumnKernelInfo* info);
