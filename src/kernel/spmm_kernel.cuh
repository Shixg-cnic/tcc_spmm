// Bounded panel tasks with async A/IDs and in-kernel final reduction.
#pragma once
#include "spmm.cuh"
#include <cstddef>

namespace {
__device__ __forceinline__ unsigned toTf32(float value) {
    unsigned result;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(result) : "f"(value));
    return result;
}
__device__ __forceinline__ void multiplyAccumulate(const unsigned (&x)[4], unsigned a0, unsigned a1,
                                                   float (&accumulators)[4]) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(accumulators[0]), "+f"(accumulators[1]), "+f"(accumulators[2]),
                   "+f"(accumulators[3])
                 : "r"(x[0]), "r"(x[1]), "r"(x[2]), "r"(x[3]), "r"(a0), "r"(a1));
}

struct FeatureFragments {
    unsigned x[2][4];
    unsigned mask0, mask1, base0, base1;
};

template <unsigned Features>
__device__ __forceinline__ void loadFeatureSources(const std::int8_t* packedX, const float* scales,
                                                   unsigned cols, unsigned featureBase,
                                                   unsigned lane, unsigned s0, unsigned s1,
                                                   FeatureFragments& tile) {
    const unsigned laneGroup = lane >> 2;
#pragma unroll
    for (unsigned part = 0; part < 2; ++part) {
#pragma unroll
        for (unsigned item = 0; item < 4; ++item)
            tile.x[part][item] = 0;
    }
    if (s0 < cols) {
        const float multiplier = scales[s0];
        const auto base = std::size_t(s0) * Features + featureBase;
#pragma unroll
        for (unsigned part = 0; part < 2; ++part) {
            const auto offset = base + part * 16 + laneGroup * 2;
            const unsigned pair = *reinterpret_cast<const unsigned short*>(packedX + offset);
            tile.x[part][0] = toTf32(float(static_cast<std::int8_t>(pair)) * multiplier);
            tile.x[part][1] = toTf32(float(static_cast<std::int8_t>(pair >> 8)) * multiplier);
        }
    }
    if (s1 < cols) {
        const float multiplier = scales[s1];
        const auto base = std::size_t(s1) * Features + featureBase;
#pragma unroll
        for (unsigned part = 0; part < 2; ++part) {
            const auto offset = base + part * 16 + laneGroup * 2;
            const unsigned pair = *reinterpret_cast<const unsigned short*>(packedX + offset);
            tile.x[part][2] = toTf32(float(static_cast<std::int8_t>(pair)) * multiplier);
            tile.x[part][3] = toTf32(float(static_cast<std::int8_t>(pair >> 8)) * multiplier);
        }
    }
}

template <unsigned Features>
__device__ __forceinline__ void
loadFeatureFragments(Sw32DeviceView matrix, unsigned panel, const std::int8_t* packedX,
                     const float* scales, unsigned cols, unsigned featureBase, unsigned lane,
                     FeatureFragments& tile) {
    const std::size_t column = std::size_t(panel) * 8 + (lane & 3);
    const unsigned s0 = matrix.sourceIds[column], s1 = matrix.sourceIds[column + 4];
    const auto meta0 = matrix.sourceMetadata[column], meta1 = matrix.sourceMetadata[column + 4];
    tile.mask0 = unsigned(meta0);
    tile.mask1 = unsigned(meta1);
    tile.base0 = unsigned(meta0 >> 32);
    tile.base1 = unsigned(meta1 >> 32);
    loadFeatureSources<Features>(packedX, scales, cols, featureBase, lane, s0, s1, tile);
}

template <unsigned RW>
__device__ __forceinline__ void accumulateRowGroup(Sw32DeviceView matrix,
                                                   const FeatureFragments& tile, unsigned laneGroup,
                                                   float (&accumulators)[2][4]) {
    const unsigned bit = 1u << (RW * 8 + laneGroup);
    unsigned a0 = 0, a1 = 0;
    if (tile.mask0 & bit)
        a0 = toTf32(matrix.values[tile.base0 + __popc(tile.mask0 & (bit - 1))]);
    if (tile.mask1 & bit)
        a1 = toTf32(matrix.values[tile.base1 + __popc(tile.mask1 & (bit - 1))]);
#pragma unroll
    for (unsigned part = 0; part < 2; ++part)
        multiplyAccumulate(tile.x[part], a0, a1, accumulators[part]);
}

__device__ __forceinline__ void copyFloatAsync(float* dst, const float* src, unsigned bytes) {
    const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::"r"(address), "l"(src), "r"(bytes)
                 : "memory");
}
__device__ __forceinline__ void copyIdsAsync(unsigned* dst, const unsigned* src) {
    const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::"r"(address), "l"(src) : "memory");
}
__device__ __forceinline__ void commitCopies() {
    asm volatile("cp.async.commit_group;" ::: "memory");
}
__device__ __forceinline__ void waitCopies() {
    asm volatile("cp.async.wait_group 0;" ::: "memory");
}
template <unsigned RW>
__device__ __forceinline__ void
stageExclusive(Sw32DeviceView matrix, unsigned batch, unsigned count, unsigned lane,
               float (&sharedA)[8][64], unsigned (&sharedSources)[8][8]) {
    const unsigned warp = threadIdx.x >> 5;
// Four producer warps stage up to eight tiles into the inactive buffer.
#pragma unroll 1
    for (unsigned producerPanel = warp; producerPanel < count; producerPanel += 4) {
        if (lane < 2)
            copyIdsAsync(&sharedSources[producerPanel][lane * 4],
                         matrix.sourceIds + std::size_t(batch + producerPanel) * 8 + lane * 4);
        const std::size_t column = std::size_t(batch + producerPanel) * 8 + (lane & 3);
        const auto meta0 = matrix.sourceMetadata[column], meta1 = matrix.sourceMetadata[column + 4];
        const unsigned mask0 = unsigned(meta0), mask1 = unsigned(meta1);
        const unsigned bit = 1u << (RW * 8 + (lane >> 2));
        const bool valid0 = (mask0 & bit) != 0, valid1 = (mask1 & bit) != 0;
        const unsigned offset0 = valid0 ? unsigned(meta0 >> 32) + __popc(mask0 & (bit - 1)) : 0u;
        const unsigned offset1 = valid1 ? unsigned(meta1 >> 32) + __popc(mask1 & (bit - 1)) : 0u;
        copyFloatAsync(&sharedA[producerPanel][lane], matrix.values + offset0, valid0 ? 4u : 0u);
        copyFloatAsync(&sharedA[producerPanel][lane + 32], matrix.values + offset1,
                       valid1 ? 4u : 0u);
    }
    commitCopies();
}
template <unsigned Features, unsigned RW>
__device__ __forceinline__ void
accumulateExclusiveBucket(Sw32DeviceView matrix, const std::int8_t* packedX, const float* scales,
                          unsigned cols, unsigned sw, unsigned taskBegin, unsigned taskEnd,
                          unsigned featureBase, unsigned lane, float (&sharedA)[2][8][64],
                          unsigned (&sharedSources)[2][8][8], float (&accumulators)[2][4]) {
    const std::size_t first = std::size_t(sw) * 17 + RW + 1;
    const unsigned begin = max(matrix.bucketOffsets[first], taskBegin),
                   end = min(matrix.bucketOffsets[first + 1], taskEnd);
    if (begin >= end)
        return;
    unsigned buffer = 0;
    stageExclusive<RW>(matrix, begin, min(8u, end - begin), lane, sharedA[buffer],
                       sharedSources[buffer]);
    waitCopies();
    __syncthreads();
    for (unsigned batch = begin; batch < end; batch += 8) {
        const unsigned next = batch + 8, count = min(8u, end - batch);
        // These copies overlap current-buffer X loads, dequantization and MMA.
        if (next < end)
            stageExclusive<RW>(matrix, next, min(8u, end - next), lane, sharedA[buffer ^ 1],
                               sharedSources[buffer ^ 1]);
        for (unsigned panel = 0; panel < count; ++panel) {
            FeatureFragments tile;
            loadFeatureSources<Features>(packedX, scales, cols, featureBase, lane,
                                         sharedSources[buffer][panel][lane & 3],
                                         sharedSources[buffer][panel][(lane & 3) + 4], tile);
            const unsigned a0 = toTf32(sharedA[buffer][panel][lane]);
            const unsigned a1 = toTf32(sharedA[buffer][panel][lane + 32]);
#pragma unroll
            for (unsigned part = 0; part < 2; ++part)
                multiplyAccumulate(tile.x[part], a0, a1, accumulators[part]);
        }
        waitCopies();
        // Publish next A/IDs and finish all reads before reusing the old buffer.
        __syncthreads();
        buffer ^= 1;
    }
}

// Bound the work assigned to one CTA without rearranging matrix data.
template <unsigned Features>
__global__ __launch_bounds__(128, 8) void spmmInt8Kernel(
    Sw32DeviceView matrix, const std::int8_t* __restrict__ packedX,
    const float* __restrict__ scales, float* __restrict__ output, unsigned rows, unsigned cols,
    const Sw32Task* tasks, volatile float* partials, const Sw32ReduceRange* reductions,
    const unsigned* partIds, unsigned* counters) {
    // Each CTA handles one panel task and 128 features; each warp owns 32 features.
    const unsigned lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const unsigned laneGroup = lane >> 2, k = lane & 3;
    const unsigned featureBase = blockIdx.y * 128 + warp * 32;
    const Sw32Task task = tasks[blockIdx.x];
    const unsigned sw = task.sw;
    const unsigned rowbase = sw * 32;
    __shared__ unsigned isLast;
    __shared__ float exclusiveA[2][8][64];
    __shared__ __align__(16) unsigned exclusiveSources[2][8][8];
    // Four RWs, two 16-feature MMA fragments, four FP32 outputs per thread.
    float accumulators[4][2][4] = {};

    // 1. Exclusive-RW panels: double-buffer A and source IDs in shared memory.
    accumulateExclusiveBucket<Features, 0>(matrix, packedX, scales, cols, sw, task.begin, task.end,
                                           featureBase, lane, exclusiveA, exclusiveSources,
                                           accumulators[0]);
    accumulateExclusiveBucket<Features, 1>(matrix, packedX, scales, cols, sw, task.begin, task.end,
                                           featureBase, lane, exclusiveA, exclusiveSources,
                                           accumulators[1]);
    accumulateExclusiveBucket<Features, 2>(matrix, packedX, scales, cols, sw, task.begin, task.end,
                                           featureBase, lane, exclusiveA, exclusiveSources,
                                           accumulators[2]);
    accumulateExclusiveBucket<Features, 3>(matrix, packedX, scales, cols, sw, task.begin, task.end,
                                           featureBase, lane, exclusiveA, exclusiveSources,
                                           accumulators[3]);

// 2. Shared-source panels: reuse each register X fragment across active RWs.
#pragma unroll 1
    for (unsigned slot = 5; slot < 16; ++slot) {
        const std::size_t first = std::size_t(sw) * 17 + slot;
        const unsigned begin = max(matrix.bucketOffsets[first], task.begin),
                       end = min(matrix.bucketOffsets[first + 1], task.end);
        if (begin >= end)
            continue;
        const unsigned mask = matrix.rowGroupMasks[begin];
        for (unsigned panel = begin; panel < end; ++panel) {
            FeatureFragments tile;
            loadFeatureFragments<Features>(matrix, panel, packedX, scales, cols, featureBase, lane,
                                           tile);
            if (mask & 1)
                accumulateRowGroup<0>(matrix, tile, laneGroup, accumulators[0]);
            if (mask & 2)
                accumulateRowGroup<1>(matrix, tile, laneGroup, accumulators[1]);
            if (mask & 4)
                accumulateRowGroup<2>(matrix, tile, laneGroup, accumulators[2]);
            if (mask & 8)
                accumulateRowGroup<3>(matrix, tile, laneGroup, accumulators[3]);
        }
    }
// 3. Write the task result: directly to Y, or to its private partial sums.
#pragma unroll
    for (unsigned rw = 0; rw < 4; ++rw) {
#pragma unroll
        for (unsigned part = 0; part < 2; ++part) {
#pragma unroll
            for (unsigned i = 0; i < 4; ++i) {
                const unsigned row = rowbase + rw * 8 + k * 2 + (i & 1);
                const unsigned col = featureBase + part * 16 + laneGroup + (i >= 2 ? 8 : 0);
                if (row < rows) {
                    if (task.partial == UINT32_MAX)
                        output[std::size_t(row) * Features + col] = accumulators[rw][part][i];
                    else if (task.mask & (1u << rw)) {
                        const std::size_t partialRw =
                            task.partial + __popc(task.mask & ((1u << rw) - 1));
                        const std::size_t offset = (partialRw * 8 + row % 8) * Features + col;
                        partials[offset] = accumulators[rw][part][i];
                    }
                }
            }
        }
    }
    // 4. Split SWs: the last CTA reduces partial sums in fixed task order.
    if (task.partial != UINT32_MAX) {
        // The barrier gathers all producer stores; the acq_rel ticket publishes them.
        __syncthreads();
        if (threadIdx.x == 0) {
            unsigned ticket;
            unsigned* counter = counters + std::size_t(sw) * (Features / 128) + blockIdx.y;
            asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                         : "=r"(ticket)
                         : "l"(counter), "r"(task.chunks - 1)
                         : "memory");
            isLast = (ticket == task.chunks - 1);
        }
        __syncthreads();
        if (isLast) {
            for (unsigned rw = 0; rw < 4; ++rw) {
                const Sw32ReduceRange reduce = reductions[task.reduceFirst + rw];
                const unsigned row = sw * 32 + rw * 8;
                if (row >= rows)
                    continue;
                const unsigned validRows = min(8u, rows - row);
                for (unsigned element = threadIdx.x; element < validRows * 128;
                     element += blockDim.x) {
                    const unsigned offset =
                        (element / 128) * Features + blockIdx.y * 128 + (element % 128);
                    float sum = 0;
                    for (unsigned item = reduce.begin; item < reduce.end; ++item)
                        sum += partials[std::size_t(partIds[item]) * 8 * Features + offset];
                    output[std::size_t(row) * Features + offset] = sum;
                }
            }
        }
    }
}

} // namespace
