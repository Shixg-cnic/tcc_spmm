#pragma once
#include "column_format.cuh"

// Current mainline: explicit original source IDs, fixed Q4, lifecycle tokens.
constexpr unsigned Q4_PREV=32, Q4_NEXT=64, Q4_INVALID=128;
struct Q4Task { unsigned begin, end, cw[4], count; };
struct Q4Format {
    unsigned rows=0, cols=0;
    std::vector<unsigned> cwCols, cwOffset{0}, tileRows, valueOffset{0};
    std::vector<std::uint64_t> bitmap;
    std::vector<float> values;
    std::vector<Q4Task> tasks;
    std::vector<unsigned> tileOrder;
    std::vector<std::uint8_t> orderCW, rowToken;
    std::uint64_t P=0, U=0, fallbackGroups=0;
    unsigned peakSlots=0;
    double groupingMs=0, formatMs=0, planMs=0;
};
Q4Format buildQ4(unsigned rows,unsigned cols,const CanonicalCSR&,bool grouped,
    std::uint64_t adjacencyBudget=262144);
struct DeviceQ4 {
    unsigned rows,cols,windows,taskCount;
    const unsigned *cwCols,*cwOffset,*tileRows,*valueOffset,*tileOrder;
    const std::uint64_t* bitmap;
    const float* values;
    const Q4Task* tasks;
    const std::uint8_t *orderCW,*rowToken;
};
enum class Q4Version { Q1, Sync, Bits, Float4, Pipeline };
cudaError_t launchQ4(const DeviceQ4&,const float*,float*,unsigned,Q4Version,
    cudaStream_t stream=nullptr,bool clear=true);
cudaError_t q4Attributes(Q4Version,ColumnKernelInfo*);
