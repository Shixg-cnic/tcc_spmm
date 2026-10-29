#include "spmm.cuh"
#include "kernel_info.cuh"
#include "spmm_kernel.cuh"

namespace {
template<unsigned Features>
cudaError_t attributes(cudaFuncAttributes* out) {
    return cudaFuncGetAttributes(out,spmmInt8Kernel<Features>);
}
template<unsigned Features>
cudaError_t occupancy(int* out) {
    return cudaOccupancyMaxActiveBlocksPerMultiprocessor(out,spmmInt8Kernel<Features>,128,0);
}
template<unsigned Features>
cudaError_t launchForFeatureDimension(Sw32DeviceView d,const std::int8_t* q,
    const float* scale,float* y,unsigned rows,unsigned cols,SpmmWorkspace workspace,cudaStream_t stream) {
    // One CTA computes at most 384 panels and 128 features; the last
    // contributor to a split SW also reduces its private partials.
    const dim3 grid(d.taskCount,Features/128);
    spmmInt8Kernel<Features><<<grid,128,0,stream>>>(d,q,scale,y,rows,cols,
        d.tasks,workspace.partials,d.reductions,d.partialIds,workspace.counters);
    return cudaGetLastError();
}
}

cudaError_t launchSpmmInt8(Sw32DeviceView d,
    const std::int8_t* q,const float* scale,float* y,
    std::uint32_t rows,std::uint32_t cols,std::uint32_t features,
    SpmmWorkspace workspace,cudaStream_t stream) {
    if (!rows || !cols || !q || !scale || !y || !d.windowOffsets || !d.bucketOffsets || !d.tasks || !d.taskCount) return cudaErrorInvalidValue;
    if (features==128) return launchForFeatureDimension<128>(d,q,scale,y,rows,cols,workspace,stream);
    if (features==256) return launchForFeatureDimension<256>(d,q,scale,y,rows,cols,workspace,stream);
    return cudaErrorInvalidValue;
}

// Optional resource queries: these functions do not launch computation.
const char* spmmKernelName() {
    return "sw32_bounded_panels_async_fused_reduce";
}
cudaError_t getSpmmKernelAttributes(cudaFuncAttributes* out,std::uint32_t features) {
    if (!out) return cudaErrorInvalidValue;
    if (features==128) return attributes<128>(out);
    if (features==256) return attributes<256>(out);
    return cudaErrorInvalidValue;
}
cudaError_t getSpmmActiveBlocksPerSm(int* out,std::uint32_t features) {
    if (!out) return cudaErrorInvalidValue;
    if (features==128) return occupancy<128>(out);
    if (features==256) return occupancy<256>(out);
    return cudaErrorInvalidValue;
}
