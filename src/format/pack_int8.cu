#include "format/int8_features.cuh"
#include <cstddef>

namespace {
__global__ void pack16(const std::int8_t* __restrict__ q,
                       unsigned short* __restrict__ out, std::size_t pairs) {
    const std::size_t pair = std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (pair < pairs) {
        const std::size_t offset = (pair>>3)*16+(pair&7);
        const unsigned low = static_cast<std::uint8_t>(q[offset]);
        const unsigned high = static_cast<std::uint8_t>(q[offset+8]);
        out[pair] = static_cast<unsigned short>(low | (high<<8));
    }
}
}

cudaError_t packFeaturesInt8x16(const std::int8_t* q, std::int8_t* output,
                  std::uint32_t rows, std::uint32_t features, cudaStream_t stream) {
    if (!q || !output || q == output || !rows || !features || features%16)
        return cudaErrorInvalidValue;
    const std::size_t pairs = std::size_t(rows)*features/2;
    const std::size_t blocks = (pairs+255)/256;
    if (blocks > 2147483647ull) return cudaErrorInvalidValue;
    pack16<<<unsigned(blocks),256,0,stream>>>(q,reinterpret_cast<unsigned short*>(output),pairs);
    return cudaGetLastError();
}
