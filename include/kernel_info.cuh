#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// Diagnostics only: none of these functions launches SpMM or chooses a faster
// configuration. The computation entry is launchSpmmInt8() in spmm.cuh.
const char* spmmKernelName();
// Compiled resource usage (e.g. numRegs, sharedSizeBytes).
cudaError_t getSpmmKernelAttributes(cudaFuncAttributes* out, std::uint32_t features);
// CUDA's theoretical number of resident 128-thread CTAs per SM, not measured
// utilization or a load-balancing decision. Supports F128/F256 only.
cudaError_t getSpmmActiveBlocksPerSm(int* out, std::uint32_t features);
