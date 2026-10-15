// Adaptive (dynamic SuperWindow) Column kernel.
//
// Each CTA reads one SWTask from the GPU task table: a variable-length
// SuperWindow (possibly a chunk of a split heavy CW). Inside a task the CW
// tile ranges come from the base colWindowOffset clamped to [tileBegin,
// tileEnd); X is loaded once per CW and reused across its tiles, exactly like
// the fixed-SW kernels. SW-local row slots + per-SW cache directory come from
// the schedule. Output is global atomicAdd, optionally staged through a
// per-SW shared cache (C3-style, 32 rows, padded stride).
#include "column_schedule.cuh"
#include "mma_tf32.cuh"
#include "pipeline.cuh"

namespace {
using U = std::uint32_t;
constexpr std::size_t CACHE_BYTES = std::size_t(COLUMN_CACHE_SIZE)*COLUMN_CACHE_STRIDE*sizeof(float);

__device__ void loadX(const float* x, U w, U cols, U f, U features, float b[2][4]) {
    U lane=threadIdx.x%32, group=lane/4, k=lane%4;
    #pragma unroll
    for (int part=0; part<2; ++part) {
        #pragma unroll
        for (int i=0; i<4; ++i) {
            U col=w*8+k+(i>=2 ? 4 : 0);
            U feature=f+group+(i%2)*8+part*16;
            b[part][i]=col<cols && feature<features ? x[std::size_t(col)*features+feature] : 0.f;
        }
    }
}
// Fused INT4 loader (see loadXInt4 in column_spmm.cu for the packing).
__device__ void loadXInt4(const DeviceInt4X* x, U w, U cols, U f, U features, float b[2][4]) {
    U lane=threadIdx.x%32, group=lane/4, k=lane%4;
    #pragma unroll
    for (int part=0; part<2; ++part) {
        #pragma unroll
        for (int i=0; i<4; ++i) {
            U col=w*8+k+(i>=2 ? 4 : 0);
            U feature=f+group+(i%2)*8+part*16;
            float v=0.f;
            if (col<cols && feature<features) {
                std::size_t idx = packedByteIndex(col, feature, features);
                v=int4ToFloat(nibbleAt(x->q,idx,feature),x->scale[col]);
            }
            b[part][i]=v;
        }
    }
}
__device__ void stageTile(DeviceColumnAdaptive m, U tile, float* a, U* slots) {
    U tid=threadIdx.x;
    if (tid<64) {
        auto bitmap=m.tileLocalBit[tile];
        a[tid]=(bitmap & (1ULL<<tid)) ?
            m.values[m.tileValueOffset[tile]+__popcll(bitmap & ((1ULL<<tid)-1))] : 0.f;
    }
    if (tid<8) slots[tid]=m.tileRowSlot[std::size_t(tile)*8+tid];
}
__device__ void prefetchTile(DeviceColumnAdaptive m, U tile, float* a, U* slots) {
    U tid=threadIdx.x;
    if (tid<64) {
        auto bitmap=m.tileLocalBit[tile];
        if (bitmap & (1ULL<<tid)) {
            auto index=m.tileValueOffset[tile]+__popcll(bitmap & ((1ULL<<tid)-1));
            async_copy(static_cast<U>(__cvta_generic_to_shared(a+tid)),m.values+index);
        } else a[tid]=0.f;
    }
    if (tid<8) async_copy_idx(static_cast<U>(__cvta_generic_to_shared(slots+tid)),
        m.tileRowSlot+std::size_t(tile)*8+tid);
    asm volatile("cp.async.commit_group;" ::: "memory");
}

template<bool UseCache, bool Int4=false>
__global__ void columnAdaptiveKernel(DeviceColumnAdaptive m, const void* x, float* y, U features) {
    __shared__ float a[2][64];
    __shared__ U slots[2][8];
    extern __shared__ float cache[];
    const SWTask task = m.tasks[blockIdx.x];
    const U rowBase = m.swRowOffset[task.swId];
    const U blockFeature = blockIdx.y*COLUMN_FEATURE_TILE;
    const U fBase = blockFeature + (threadIdx.x/32)*32;
    if constexpr(UseCache) {
        const U count = m.swCachedCount[task.swId];
        for (U i=threadIdx.x;i<count*COLUMN_FEATURE_TILE;i+=blockDim.x)
            cache[(i/COLUMN_FEATURE_TILE)*COLUMN_CACHE_STRIDE+i%COLUMN_FEATURE_TILE]=0.f;
        __syncthreads();
    }
    float b[2][4];
    for (U cw=task.cwBegin; cw<task.cwEnd; ++cw) {
        U cwFirst = max(task.tileBegin, m.colWindowOffset[cw]);
        U cwLast  = min(task.tileEnd, m.colWindowOffset[cw+1]);
        if (cwFirst>=cwLast) continue;
        if constexpr (Int4) loadXInt4(static_cast<const DeviceInt4X*>(x), cw, m.cols, fBase, features, b);
        else loadX(static_cast<const float*>(x), cw, m.cols, fBase, features, b);
        stageTile(m, cwFirst, a[0], slots[0]);
        __syncthreads();
        for (U tile=cwFirst; tile<cwLast; ++tile) {
            const U buf = (tile-cwFirst)&1;
            if (tile+1<cwLast) prefetchTile(m, tile+1, a[buf^1], slots[buf^1]);
            const U lane=threadIdx.x%32, group=lane/4, k=lane%4;
            const float* av=a[buf];
            const U* sv=slots[buf];
            float sparse[2]={av[group*8+k], av[group*8+k+4]};
            #pragma unroll
            for (int part=0; part<2; ++part) {
                float c[4]={0.f,0.f,0.f,0.f};
                tf32_m16n8k8(b[part],sparse,c);
                #pragma unroll
                for (int i=0;i<4;++i) {
                    U slot=sv[k*2+i%2];
                    U feature=fBase+group+(i>=2 ? 8 : 0)+part*16;
                    if (slot!=UINT32_MAX && feature<features) {
                        if constexpr(UseCache) {
                            U cached=m.swRowCacheSlot[rowBase+slot];
                            if (cached!=UINT32_MAX) {
                                cache[cached*COLUMN_CACHE_STRIDE+(feature-blockFeature)]+=c[i];
                                continue;
                            }
                        }
                        U row=m.swRows[rowBase+slot];
                        atomicAdd(y+std::size_t(row)*features+feature,c[i]);
                    }
                }
            }
            asm volatile("cp.async.wait_group 0;" ::: "memory");
            __syncthreads();
        }
    }
    if constexpr(UseCache) {
        const U count=m.swCachedCount[task.swId];
        __syncthreads();
        for (U i=threadIdx.x;i<count*COLUMN_FEATURE_TILE;i+=blockDim.x) {
            U feature=blockFeature+i%COLUMN_FEATURE_TILE;
            if (feature<features) {
                U slot=m.swCachedRowSlot[std::size_t(task.swId)*COLUMN_CACHE_SIZE+i/COLUMN_FEATURE_TILE];
                U row=m.swRows[rowBase+slot];
                atomicAdd(y+std::size_t(row)*features+feature,
                    cache[(i/COLUMN_FEATURE_TILE)*COLUMN_CACHE_STRIDE+i%COLUMN_FEATURE_TILE]);
            }
        }
    }
}
} // namespace

cudaError_t launchColumnAdaptiveSpMM(const DeviceColumnAdaptive& matrix, const float* x, float* y,
    std::uint32_t features, bool useCache, cudaStream_t stream, bool clearOutput) {
    if (!matrix.taskCount || !features || !x || !y) return cudaErrorInvalidValue;
    if (clearOutput) {
        auto err = cudaMemsetAsync(y, 0, std::size_t(matrix.rows)*features*sizeof(float), stream);
        if (err) return err;
    }
    dim3 grid(matrix.taskCount, (std::uint64_t(features)+COLUMN_FEATURE_TILE-1)/COLUMN_FEATURE_TILE);
    if (useCache)
        columnAdaptiveKernel<true><<<grid, COLUMN_FEATURE_TILE, CACHE_BYTES, stream>>>(matrix, x, y, features);
    else
        columnAdaptiveKernel<false><<<grid, COLUMN_FEATURE_TILE, 0, stream>>>(matrix, x, y, features);
    return cudaGetLastError();
}

cudaError_t launchColumnAdaptiveInt4SpMM(const DeviceColumnAdaptive& matrix, const DeviceInt4X& x, float* y,
    std::uint32_t features, bool useCache, cudaStream_t stream, bool clearOutput) {
    if (!matrix.taskCount || !features || !x.q || !x.scale || !y) return cudaErrorInvalidValue;
    if (clearOutput) {
        auto err = cudaMemsetAsync(y, 0, std::size_t(matrix.rows)*features*sizeof(float), stream);
        if (err) return err;
    }
    dim3 grid(matrix.taskCount, (std::uint64_t(features)+COLUMN_FEATURE_TILE-1)/COLUMN_FEATURE_TILE);
    if (useCache)
        columnAdaptiveKernel<true,true><<<grid, COLUMN_FEATURE_TILE, CACHE_BYTES, stream>>>(matrix, &x, y, features);
    else
        columnAdaptiveKernel<false,true><<<grid, COLUMN_FEATURE_TILE, 0, stream>>>(matrix, &x, y, features);
    return cudaGetLastError();
}
