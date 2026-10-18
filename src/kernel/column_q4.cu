// MMA lane mapping and cp.async helpers derived from Acc-SpMM,
// MIT (c) 2025 Haisha Zhao; Q4 scheduling and lifetime cache implemented here.
#include "column_q4.cuh"
#include "mma_tf32.cuh"
#include "pipeline.cuh"
namespace {
using U=unsigned;
struct Fragment { U bits[2][4]; };
template<bool Convert> __device__ __forceinline__ Fragment loadX(DeviceQ4 m,const float* x,U cw,U f,U F,bool valid) {
    Fragment out{};U g=(threadIdx.x%32)/4,k=threadIdx.x%4;
    #pragma unroll
    for(int p=0;p<2;++p) {
        #pragma unroll
        for(int i=0;i<4;++i) {
            U src=valid?m.cwCols[std::size_t(cw)*8+k+(i>=2?4:0)]:UINT32_MAX;
            float v=src<m.cols?x[std::size_t(src)*F+f+g+(i%2)*8+p*16]:0;
            out.bits[p][i]=Convert?toTf32RnaBits(v):__float_as_uint(v);
        }
    }
    return out;
}
template<bool Async,bool XorA> __device__ __forceinline__ void stage(DeviceQ4 m,U tile,float* a) {
    U t=threadIdx.x;
    U physical=XorA?(t^((t>>3)&4)):t;
    if(t<64) {
        auto bits=m.bitmap[tile];
        if(bits&(1ULL<<t)) {
            auto ptr=m.values+m.valueOffset[tile]+__popcll(bits&((1ULL<<t)-1));
            if constexpr(Async) async_copy(U(__cvta_generic_to_shared(a+physical)),ptr);
            else a[physical]=*ptr;
        } else a[physical]=0;
    }
    if constexpr(Async) asm volatile("cp.async.commit_group;" ::: "memory");
}
template<bool Q1,bool Convert,bool Wide,bool Pipeline,unsigned H,bool XorA> __global__ void kernel(DeviceQ4 m,const float* x,float* y,U F) {
    __shared__ float a[Pipeline?2:1][64];
    // H changes physical capacity only; token slot decoding remains &31.
    __shared__ float cache[Q1?1:H][Q1?1:132];
    U f=blockIdx.y*128+(threadIdx.x/32)*32,g=(threadIdx.x%32)/4,k=threadIdx.x%4;
    Q4Task task{};
    if constexpr(Q1) {task.cw[0]=blockIdx.x;task.count=1;task.begin=m.cwOffset[blockIdx.x];task.end=m.cwOffset[blockIdx.x+1];}
    else task=m.tasks[blockIdx.x];
    Fragment x0=loadX<Convert>(m,x,task.cw[0],f,F,true);
    Fragment x1{},x2{},x3{};
    if constexpr(!Q1) {
        x1=loadX<Convert>(m,x,task.cw[1],f,F,task.count>1);
        x2=loadX<Convert>(m,x,task.cw[2],f,F,task.count>2);
        x3=loadX<Convert>(m,x,task.cw[3],f,F,task.count>3);
    }
    if constexpr(Pipeline) {
        if(task.begin<task.end) stage<false,XorA>(m,m.tileOrder[task.begin],a[0]);
        __syncthreads();
    }
    for(U o=task.begin;o<task.end;++o) {
        U tile=Q1?o:m.tileOrder[o],buf=Pipeline?((o-task.begin)&1):0;
        if constexpr(Pipeline) {if(o+1<task.end) stage<true,XorA>(m,m.tileOrder[o+1],a[buf^1]);}
        else {stage<false,XorA>(m,tile,a[0]);__syncthreads();}
        U cw=Q1?0:m.orderCW[o];
        U swizzle=XorA?(g&4):0;
        float a0=a[buf][g*8+(k^swizzle)],a1=a[buf][g*8+((k+4)^swizzle)];
        U sparse[2]={toTf32RnaBits(a0),toTf32RnaBits(a1)};
        #pragma unroll
        for(int part=0;part<2;++part) {
            float c[4]; U tokens[2];
            #pragma unroll
            for(int j=0;j<2;++j) tokens[j]=Q1?(m.tileRows[std::size_t(tile)*8+k*2+j]==UINT32_MAX?Q4_INVALID:0):m.rowToken[std::size_t(o)*8+k*2+j];
            #pragma unroll
            for(int i=0;i<4;++i) {
                U token=tokens[i%2],feature=(threadIdx.x/32)*32+g+(i>=2?8:0)+part*16;
                c[i]=!Q1&&(token&Q4_PREV)?cache[token&31][feature]:0;
            }
            if constexpr(Convert) {
                switch(cw) {
                    case 0:mmaTf32Bits(x0.bits[part],sparse,c);break;
                    case 1:mmaTf32Bits(x1.bits[part],sparse,c);break;
                    case 2:mmaTf32Bits(x2.bits[part],sparse,c);break;
                    default:mmaTf32Bits(x3.bits[part],sparse,c);break;
                }
            } else {
                float b[4],s[2]={a0,a1};
                #pragma unroll
                for(int i=0;i<4;++i) b[i]=__uint_as_float(cw==0?x0.bits[part][i]:cw==1?x1.bits[part][i]:cw==2?x2.bits[part][i]:x3.bits[part][i]);
                tf32_m16n8k8(b,s,c);
            }
            #pragma unroll
            for(int i=0;i<4;++i) {
                U token=tokens[i%2],feature=f+g+(i>=2?8:0)+part*16;
                if(token&Q4_INVALID) continue;
                if(!Q1&&(token&Q4_NEXT)) cache[token&31][feature-blockIdx.y*128]=c[i];
                else if constexpr(!Wide) {
                    U row=m.tileRows[std::size_t(tile)*8+k*2+i%2];
                    atomicAdd(y+std::size_t(row)*F+feature,c[i]);
                }
            }
            if constexpr(Wide) {
                float v[4];U quad=4*(g&3);
                // Every lane completes all shuffles before state-dependent output.
                #pragma unroll
                for(int j=0;j<4;++j) {
                    U q=quad+j,src=(q&7)*4+k;
                    float a0=__shfl_sync(0xffffffff,c[0],src),a1=__shfl_sync(0xffffffff,c[1],src);
                    float a2=__shfl_sync(0xffffffff,c[2],src),a3=__shfl_sync(0xffffffff,c[3],src);
                    v[j]=g<4?(q<8?a0:a2):(q<8?a1:a3);
                }
                U token=tokens[g>=4];
                if(!(token&(Q4_INVALID|Q4_NEXT))) {
                    U row=m.tileRows[std::size_t(tile)*8+k*2+(g>=4)];
                    #if __CUDA_ARCH__ >= 900
                    atomicAdd(reinterpret_cast<float4*>(y+std::size_t(row)*F+f+quad+part*16),make_float4(v[0],v[1],v[2],v[3]));
                    #else
                    for(int j=0;j<4;++j) atomicAdd(y+std::size_t(row)*F+f+quad+part*16+j,v[j]);
                    #endif
                }
            }
        }
        if constexpr(Pipeline) asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
    }
}

// Independent compile-time switches: asynchronous A never implies float4 output.
template<unsigned H,bool XorA> const void* pointer(Q4Version v) {
    switch(v) {
        case Q4Version::Q1:return (const void*)kernel<true,true,false,false,32,false>;
        case Q4Version::Sync:return (const void*)kernel<false,false,false,false,H,XorA>;
        case Q4Version::Bits:return (const void*)kernel<false,true,false,false,H,XorA>;
        case Q4Version::Float4:return (const void*)kernel<false,true,true,false,H,XorA>;
        case Q4Version::Pipeline:return (const void*)kernel<false,true,true,true,H,XorA>;
        case Q4Version::ScalarPipeline:return (const void*)kernel<false,true,false,true,H,XorA>;
    }
    return nullptr;
}
const void* pointer(Q4Config c,unsigned peak) {
    if(c.version==Q4Version::Q1) return pointer<32,false>(c.version);
    unsigned h=c.cacheSlots?c.cacheSlots:peak<=16?16:peak<=24?24:32;
    if(peak>h||peak>32) return nullptr;
    switch(h) {
        case 16:return c.xorA?pointer<16,true>(c.version):pointer<16,false>(c.version);
        case 24:return c.xorA?pointer<24,true>(c.version):pointer<24,false>(c.version);
        case 32:return c.xorA?pointer<32,true>(c.version):pointer<32,false>(c.version);
    }
    return nullptr;
}
}
cudaError_t launchQ4(const DeviceQ4& m,const float* x,float* y,U F,Q4Config config,cudaStream_t stream,bool clear) {
    auto p=pointer(config,m.peakSlots);
    if(!m.rows||!m.cols||!x||!y||(!F||F>512||F%128)||!p) return cudaErrorInvalidValue;
    if(clear) {auto err=cudaMemsetAsync(y,0,std::size_t(m.rows)*F*sizeof(float),stream);if(err!=cudaSuccess)return err;}
    U count=config.version==Q4Version::Q1?m.windows:m.taskCount;
    if(!count) return cudaSuccess;
    void* args[]={const_cast<DeviceQ4*>(&m),&x,&y,&F};
    return cudaLaunchKernel(p,dim3(count,F/128),dim3(128),args,0,stream);
}
cudaError_t q4Attributes(Q4Config config,unsigned peak,ColumnKernelInfo* info) {
    auto p=pointer(config,peak);if(!p||!info)return cudaErrorInvalidValue;
    cudaFuncAttributes a;auto e=cudaFuncGetAttributes(&a,p);if(e!=cudaSuccess)return e;
    info->registers=a.numRegs;info->sharedBytes=a.sharedSizeBytes;info->localBytes=a.localSizeBytes;
    return cudaOccupancyMaxActiveBlocksPerMultiprocessor(&info->activeBlocksPerSM,p,128,0);
}
// Preserve round-one API semantics (H32 and linear A layout).
cudaError_t launchQ4(const DeviceQ4& m,const float* x,float* y,U f,Q4Version v,cudaStream_t stream,bool clear) {
    return launchQ4(m,x,y,f,Q4Config{v,32,false},stream,clear);
}
cudaError_t q4Attributes(Q4Version v,ColumnKernelInfo* info) {
    return q4Attributes(Q4Config{v,32,false},0,info);
}
