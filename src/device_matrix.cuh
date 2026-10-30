#pragma once

#include "cuda_support.cuh"
#include "spmm.cuh"

// Own the uploaded SW32 arrays and the workspace used by split tasks.
// The kernel receives only the lightweight view() and workspace() descriptors.
struct DeviceMatrix {
    DeviceBuffer<std::uint32_t> offsets, sourceIds, masks, buckets, partialIds;
    DeviceBuffer<Sw32Task> tasks;
    DeviceBuffer<Sw32ReduceRange> reductions;
    DeviceBuffer<float> partials;
    DeviceBuffer<std::uint32_t> counters;
    DeviceBuffer<std::uint64_t> metadata;
    DeviceBuffer<float> values;
    std::uint32_t taskCount;

    DeviceMatrix(const Sw32HostFormat& matrix, unsigned features)
        : offsets(matrix.windowOffsets), sourceIds(matrix.sourceIds), masks(matrix.rowGroupMasks),
          buckets(matrix.bucketOffsets), partialIds(matrix.partialIds), tasks(matrix.tasks),
          reductions(matrix.reductions), partials(std::size_t(matrix.partialCount) * 8 * features),
          counters(matrix.partialCount ? (matrix.windowOffsets.size() - 1) * (features / 128) : 0),
          metadata(matrix.sourceMetadata), values(matrix.values), taskCount(matrix.tasks.size()) {
        if (matrix.partialCount) {
            const auto count = (matrix.windowOffsets.size() - 1) * (features / 128);
            checkCuda(cudaMemset(counters.ptr, 0, count * sizeof(std::uint32_t)));
        }
    }

    Sw32DeviceView view() const {
        return {offsets.ptr, sourceIds.ptr, masks.ptr,      metadata.ptr,   values.ptr,
                buckets.ptr, tasks.ptr,     reductions.ptr, partialIds.ptr, taskCount};
    }

    SpmmWorkspace workspace() const {
        return {partials.ptr, counters.ptr};
    }
};
