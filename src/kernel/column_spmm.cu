// MMA mapping and asynchronous staging derived from Acc-SpMM,
// MIT (c) 2025 Haisha Zhao. Column scheduling/output ownership are adapted here.
#include "column_format.cuh"
#include "int4_format.cuh"
#include "mma_tf32.cuh"
#include "pipeline.cuh"
#include <limits>

namespace {
using U = std::uint32_t;
constexpr int WARPS = COLUMN_FEATURE_TILE/32;

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

// Fused INT4 loader: reads the packed stream, dequantises q*s in registers,
// producing exactly the FP32 fragment the MMA consumes. Same indexing as
// loadX (feature j of node col lives at q[col*features+j], nibble j&1).
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

__device__ void stageTile(DeviceColumnSWTCF m, U tile, float* a, U* slots) {
    U tid=threadIdx.x;
    if (tid<64) {
        auto bitmap=m.tileLocalBit[tile];
        a[tid]=(bitmap & (1ULL<<tid)) ?
            m.values[m.tileValueOffset[tile]+__popcll(bitmap & ((1ULL<<tid)-1))] : 0.f;
    }
    if (tid<8) slots[tid]=m.tileRowSlot[std::size_t(tile)*8+tid];
}

// Stage the entire next tile into the alternate buffer before current MMA.
// cp.async completion is deliberately postponed until after MMA/output updates.
__device__ void prefetchTile(DeviceColumnSWTCF m, U tile, float* a, U* slots) {
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

// All warps compute disjoint feature positions. Within each window a target
// row occurs in exactly one tile, so cached updates have a unique lane owner.
// CTA barriers separate tiles/windows; no shared atomic or warp-group copies
// are necessary with this single-group scheduling.
template<int Version, bool Int4=false>
__global__ void columnKernel(DeviceColumnSWTCF m, const void* x, float* y, U features) {
    __shared__ float a[Version==0 ? 1 : 2][64];
    __shared__ U slots[Version==0 ? 1 : 2][8];
    extern __shared__ float cache[];
    const U windows=(std::uint64_t(m.cols)+7)/8;
    U firstWindow=Version>=2 ? blockIdx.x*m.superWindowSize : blockIdx.x;
    U endWindow=Version>=2 ? U(min(std::uint64_t(firstWindow)+m.superWindowSize,std::uint64_t(windows))) : firstWindow+1;
    U sw=firstWindow/m.superWindowSize;
    U rowBase=m.superWindowRowOffset[sw];
    U blockFeature=blockIdx.y*COLUMN_FEATURE_TILE;
    U fBase=blockFeature+(threadIdx.x/32)*32;
    if constexpr(Version==3) {
        for(U i=threadIdx.x;i<COLUMN_CACHE_SIZE*COLUMN_FEATURE_TILE;i+=blockDim.x)
            cache[(i/COLUMN_FEATURE_TILE)*COLUMN_CACHE_STRIDE+i%COLUMN_FEATURE_TILE]=0.f;
        __syncthreads();
    }
    float b[2][4],nextX[2][4];
    bool prefetched=false;
    for(U w=firstWindow;w<endWindow;++w) {
        if(prefetched) {
            #pragma unroll
            for(int part=0;part<2;++part) {
                #pragma unroll
                for(int i=0;i<4;++i) b[part][i]=nextX[part][i];
            }
        } else if constexpr (Int4) loadXInt4(static_cast<const DeviceInt4X*>(x),w,m.cols,fBase,features,b);
        else loadX(static_cast<const float*>(x),w,m.cols,fBase,features,b);
        prefetched=false;
        U first=m.colWindowOffset[w], end=m.colWindowOffset[w+1];
        if constexpr(Version>0) {
            if(first<end) stageTile(m,first,a[0],slots[0]);
            __syncthreads();
        }
        for(U tile=first;tile<end;++tile) {
            U buffer=Version==0 ? 0 : (tile-first)%2;
            if constexpr(Version==0) {stageTile(m,tile,a[0],slots[0]); __syncthreads();}
            else if(tile+1<end) prefetchTile(m,tile+1,a[buffer^1],slots[buffer^1]);
            if constexpr(Version>=2) {
                if(tile+1==end && w+1<endWindow) {
                    if constexpr (Int4) loadXInt4(static_cast<const DeviceInt4X*>(x),w+1,m.cols,fBase,features,nextX);
                    else loadX(static_cast<const float*>(x),w+1,m.cols,fBase,features,nextX);
                    prefetched=true;
                }
            }
            U lane=threadIdx.x%32, group=lane/4, k=lane%4;
            float sparse[2]={a[buffer][group*8+k],a[buffer][group*8+k+4]};
            #pragma unroll
            for(int part=0;part<2;++part) {
                float c[4]={0.f,0.f,0.f,0.f};
                tf32_m16n8k8(b[part],sparse,c);
                #pragma unroll
                for(int i=0;i<4;++i) {
                    U slot=slots[buffer][k*2+i%2];
                    U feature=fBase+group+(i>=2 ? 8 : 0)+part*16;
                    if(slot!=UINT32_MAX && feature<features) {
                        if constexpr(Version==3) {
                            U cached=m.superWindowRowCacheSlot[rowBase+slot];
                            if(cached!=UINT32_MAX) {
                                cache[cached*COLUMN_CACHE_STRIDE+(feature-blockFeature)]+=c[i];
                                continue;
                            }
                        }
                        U row=m.superWindowRows[rowBase+slot];
                        atomicAdd(y+std::size_t(row)*features+feature,c[i]);
                    }
                }
            }
            if constexpr(Version>0) asm volatile("cp.async.wait_group 0;" ::: "memory");
            __syncthreads();
        }
    }
    if constexpr(Version==3) {
        __syncthreads();
        U count=m.superWindowCachedCount[sw];
        for(U i=threadIdx.x;i<count*COLUMN_FEATURE_TILE;i+=blockDim.x) {
            U feature=blockFeature+i%COLUMN_FEATURE_TILE;
            if(feature<features) {
                U slot=m.superWindowCachedRowSlot[std::size_t(sw)*COLUMN_CACHE_SIZE+i/COLUMN_FEATURE_TILE];
                U row=m.superWindowRows[rowBase+slot];
                atomicAdd(y+std::size_t(row)*features+feature,
                    cache[(i/COLUMN_FEATURE_TILE)*COLUMN_CACHE_STRIDE+i%COLUMN_FEATURE_TILE]);
            }
        }
    }
}

const void* kernelPointer(ColumnVersion version) {
    switch(version) {
        case ColumnVersion::C0:return reinterpret_cast<const void*>(columnKernel<0,false>);
        case ColumnVersion::C1:return reinterpret_cast<const void*>(columnKernel<1,false>);
        case ColumnVersion::C2:return reinterpret_cast<const void*>(columnKernel<2,false>);
        case ColumnVersion::C3:return reinterpret_cast<const void*>(columnKernel<3,false>);
    }
    return nullptr;
}
const void* int4KernelPointer(ColumnVersion version) {
    switch(version) {
        case ColumnVersion::C0:return reinterpret_cast<const void*>(columnKernel<0,true>);
        case ColumnVersion::C1:return reinterpret_cast<const void*>(columnKernel<1,true>);
        case ColumnVersion::C2:return reinterpret_cast<const void*>(columnKernel<2,true>);
        case ColumnVersion::C3:return reinterpret_cast<const void*>(columnKernel<3,true>);
    }
    return nullptr;
}
constexpr std::size_t CACHE_BYTES=std::size_t(COLUMN_CACHE_SIZE)*COLUMN_CACHE_STRIDE*sizeof(float);

}

cudaError_t launchColumnSpMM(const DeviceColumnSWTCF& m, const float* x, float* y,
    U features, ColumnVersion version, cudaStream_t stream, bool clearOutput) {
    if(!m.rows || !m.cols || !m.superWindowSize || !features || !x || !y || !kernelPointer(version))
        return cudaErrorInvalidValue;
    if(std::uint64_t(m.rows)*features>std::numeric_limits<std::size_t>::max()/sizeof(float) ||
        (std::uint64_t(features)+COLUMN_FEATURE_TILE-1)/COLUMN_FEATURE_TILE>65535)
        return cudaErrorInvalidValue;
    if(clearOutput) {
        auto status=cudaMemsetAsync(y,0,std::size_t(m.rows)*features*sizeof(float),stream);
        if(status!=cudaSuccess) return status;
    }
    U windows=(std::uint64_t(m.cols)+7)/8;
    U tasks=int(version)>=2 ? (std::uint64_t(windows)+m.superWindowSize-1)/m.superWindowSize : windows;
    dim3 grid(tasks,(std::uint64_t(features)+COLUMN_FEATURE_TILE-1)/COLUMN_FEATURE_TILE);
    switch(version) {
        case ColumnVersion::C0:columnKernel<0><<<grid,WARPS*32,0,stream>>>(m,x,y,features);break;
        case ColumnVersion::C1:columnKernel<1><<<grid,WARPS*32,0,stream>>>(m,x,y,features);break;
        case ColumnVersion::C2:columnKernel<2><<<grid,WARPS*32,0,stream>>>(m,x,y,features);break;
        case ColumnVersion::C3:columnKernel<3><<<grid,WARPS*32,CACHE_BYTES,stream>>>(m,x,y,features);break;
    }
    return cudaGetLastError();
}
cudaError_t launchColumnInt4SpMM(const DeviceColumnSWTCF& m, const DeviceInt4X& x, float* y,
    U features, ColumnVersion version, cudaStream_t stream, bool clearOutput) {
    if(!m.rows || !m.cols || !m.superWindowSize || !features || !x.q || !x.scale || !y ||
        !int4KernelPointer(version))
        return cudaErrorInvalidValue;
    if(std::uint64_t(m.rows)*features>std::numeric_limits<std::size_t>::max()/sizeof(float) ||
        (std::uint64_t(features)+COLUMN_FEATURE_TILE-1)/COLUMN_FEATURE_TILE>65535)
        return cudaErrorInvalidValue;
    if(clearOutput) {
        auto status=cudaMemsetAsync(y,0,std::size_t(m.rows)*features*sizeof(float),stream);
        if(status!=cudaSuccess) return status;
    }
    U windows=(std::uint64_t(m.cols)+7)/8;
    U tasks=int(version)>=2 ? (std::uint64_t(windows)+m.superWindowSize-1)/m.superWindowSize : windows;
    dim3 grid(tasks,(std::uint64_t(features)+COLUMN_FEATURE_TILE-1)/COLUMN_FEATURE_TILE);
    switch(version) {
        case ColumnVersion::C0:columnKernel<0,true><<<grid,WARPS*32,0,stream>>>(m,&x,y,features);break;
        case ColumnVersion::C1:columnKernel<1,true><<<grid,WARPS*32,0,stream>>>(m,&x,y,features);break;
        case ColumnVersion::C2:columnKernel<2,true><<<grid,WARPS*32,0,stream>>>(m,&x,y,features);break;
        case ColumnVersion::C3:columnKernel<3,true><<<grid,WARPS*32,CACHE_BYTES,stream>>>(m,&x,y,features);break;
    }
    return cudaGetLastError();
}
cudaError_t columnKernelAttributes(ColumnVersion version, ColumnKernelInfo* info) {
    auto kernel=kernelPointer(version);
    if(!kernel || !info) return cudaErrorInvalidValue;
    cudaFuncAttributes a;
    auto status=cudaFuncGetAttributes(&a,kernel);
    if(status!=cudaSuccess) return status;
    auto dynamic=version==ColumnVersion::C3 ? CACHE_BYTES : 0;
    info->registers=a.numRegs; info->sharedBytes=a.sharedSizeBytes+dynamic; info->localBytes=a.localSizeBytes;
    return cudaOccupancyMaxActiveBlocksPerMultiprocessor(&info->activeBlocksPerSM,kernel,WARPS*32,dynamic);
}
