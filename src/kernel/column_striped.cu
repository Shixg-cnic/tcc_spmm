#include "column_striped.cuh"
#include "mma_tf32.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <stdexcept>
#include <limits>

namespace {
using U = std::uint32_t;
U checked(std::size_t n) {
    if (n >= UINT32_MAX) throw std::overflow_error("striped format: 32-bit offset overflow");
    return U(n);
}
unsigned pop(std::uint64_t x) { return __builtin_popcountll(x); }

template<bool Wide4>
__global__ void columnStripedKernel(DeviceColumnStriped m, const float* x, float* y,
                                    U f, U stripe) {
    __shared__ float a[64];
    __shared__ U rows[8];
    U cw = blockIdx.x, fb = blockIdx.y * 128 + (threadIdx.x / 32) * 32;
    U lane = threadIdx.x & 31, g = lane / 4, k = lane & 3;
    U key = stripe * (m.cws + 1) + cw;
    U first = m.stripeCwOffset[key], last = m.stripeCwOffset[key + 1];
    if (first == last) return;
    float b[2][4];
    #pragma unroll
    for (int part=0; part<2; ++part) {
        #pragma unroll
        for(int i=0;i<4;++i) {
            U c = cw*8 + k + (i>=2 ? 4 : 0);
            U feat = fb+g+(i%2)*8+part*16;
            b[part][i] = c<m.cols && feat<f ? x[std::size_t(c)*f+feat] : 0.f;
        }
    }
    for(U t=first;t<last;++t) {
        if(threadIdx.x<64) {
            auto mask=m.bitmap[t];
            a[threadIdx.x] = (mask&(1ULL<<threadIdx.x))
                ? m.values[m.valueOffset[t]+__popcll(mask&((1ULL<<threadIdx.x)-1))]
                : 0.f;
        }
        if(threadIdx.x<8) rows[threadIdx.x]=m.tileRows[std::size_t(t)*8+threadIdx.x];
        __syncthreads();
        float sparse[2]={a[g*8+k],a[g*8+k+4]};
        #pragma unroll
        for(int part=0;part<2;++part) {
            float c[4]={0.f,0.f,0.f,0.f};
            tf32_m16n8k8(b[part],sparse,c);
            if constexpr(!Wide4) {
                #pragma unroll
                for(int j=0;j<4;++j) {
                    U row=rows[2*k+(j&1)];
                    U feature=fb+g+(j>=2 ? 8 : 0)+part*16;
                    if(row!=UINT32_MAX && feature<f)
                        atomicAdd(y+std::size_t(row)*f+feature,c[j]);
                }
            } else {
                U row=rows[2*k+(g>=4?1:0)];
                U quad=4*(g&3), feature=fb+quad+part*16;
                bool even=g<4;
                float v[4];
                #pragma unroll
                for(int j=0;j<4;++j) {
                    U q=quad+j, src=(q&7)*4+k;
                    float z0=__shfl_sync(0xffffffffu,c[0],src);
                    float z1=__shfl_sync(0xffffffffu,c[1],src);
                    float z2=__shfl_sync(0xffffffffu,c[2],src);
                    float z3=__shfl_sync(0xffffffffu,c[3],src);
                    v[j]=even ? (q<8?z0:z2) : (q<8?z1:z3);
                }
                if(row!=UINT32_MAX && feature+3<f)
                    atomicAdd(reinterpret_cast<float4*>(y+std::size_t(row)*f+feature),
                        make_float4(v[0],v[1],v[2],v[3]));
            }
        }
        __syncthreads();
    }
}
}

ColumnStriped makeColumnStriped(const ColumnBaseFormat& src, U stripeRows) {
    if(!src.rows || !src.cols || !stripeRows || src.colWindowOffset.size()!=std::size_t((std::uint64_t(src.cols)+7)/8)+1 ||
       src.tileRows.size()!=src.tileLocalBit.size()*8 ||
       src.tileValueOffset.size()!=src.tileLocalBit.size()+1)
        throw std::invalid_argument("invalid column base for striped conversion");
    ColumnStriped out;
    out.rows=src.rows;out.cols=src.cols;out.stripeRows=stripeRows;
    const U S=out.stripes(), W=out.cws();
    out.stripeCwOffset.reserve(std::size_t(S)*(W+1));
    out.valueOffset.push_back(0);
    // Offline prototype: scan all source tiles for each stripe. No graph reorder.
    // A future optimized converter can pre-index tiles by row band.
    for(U s=0;s<S;++s) {
        const std::uint64_t lo=std::uint64_t(s)*stripeRows;
        const std::uint64_t hi=std::min<std::uint64_t>(src.rows,lo+stripeRows);
        for(U cw=0;cw<W;++cw) {
            out.stripeCwOffset.push_back(checked(out.bitmap.size()));
            std::uint64_t bits=0;
            U occupied=0;
            U groupRows[8];
            float groupVals[64];
            U groupValCount=0;
            auto flush=[&]() {
                if(!occupied) return;
                out.bitmap.push_back(bits);
                for(U j=0;j<occupied;++j) out.tileRows.push_back(groupRows[j]);
                for(U j=occupied;j<8;++j) out.tileRows.push_back(UINT32_MAX);
                out.values.insert(out.values.end(),groupVals,groupVals+groupValCount);
                out.valueOffset.push_back(checked(out.values.size()));
                bits=0;occupied=0;groupValCount=0;
            };
            for(U tile=src.colWindowOffset[cw];tile<src.colWindowOffset[cw+1];++tile) {
                auto bitmap=src.tileLocalBit[tile];
                for(U r=0;r<8;++r) {
                    U row=src.tileRows[std::size_t(tile)*8+r];
                    if(row==UINT32_MAX || row<lo || row>=hi) continue;
                    U mask=(bitmap>>(8*r))&255u;
                    if(!mask) continue;
                    if(occupied==8) flush();
                    groupRows[occupied]=row;
                    bits |= std::uint64_t(mask) << (occupied*8);
                    std::size_t begin=src.tileValueOffset[tile]+pop(bitmap & ((std::uint64_t(1)<<(8*r))-1));
                    for(U c=0;c<8;++c) if(mask&(1u<<c)) groupVals[groupValCount++]=src.values[begin++];
                    ++occupied;
                }
            }
            flush();
        }
        out.stripeCwOffset.push_back(checked(out.bitmap.size()));
    }
    if(out.stripeCwOffset.size()!=std::size_t(S)*(W+1))
        throw std::runtime_error("stripes offsets mismatch");
    return out;
}

cudaError_t launchColumnStriped(const DeviceColumnStriped& m, const float* x,
                                float* y, U features, bool wide4, cudaStream_t stream,
                                bool clearOutput) {
    if(!m.stripeRows || !m.stripes || !m.cws || !m.stripeCwOffset || !m.tileRows ||
       !m.valueOffset || !m.bitmap || !m.values || !x || !y ||
       (features!=128 && features!=256))
        return cudaErrorInvalidValue;
    if(clearOutput) {
        auto err=cudaMemsetAsync(y,0,std::size_t(m.rows)*features*sizeof(float),stream);
        if(err!=cudaSuccess) return err;
    }
    dim3 grid(m.cws,features/128);
    for(U s=0;s<m.stripes;++s) {
        if(wide4)
            columnStripedKernel<true><<<grid,128,0,stream>>>(m,x,y,features,s);
        else
            columnStripedKernel<false><<<grid,128,0,stream>>>(m,x,y,features,s);
        auto status=cudaGetLastError();
        if(status!=cudaSuccess) return status;
    }
    return cudaSuccess;
}
