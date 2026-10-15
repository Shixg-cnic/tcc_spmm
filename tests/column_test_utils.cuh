#pragma once
#include "column_format.cuh"
#include "row_format.cuh"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>

namespace test {
inline void check(cudaError_t error) {
    if (error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
inline void require(bool condition, const char* error) {
    if (!condition) throw std::runtime_error(error);
}
using Clock=std::chrono::steady_clock;
inline double elapsed(Clock::time_point start) { return std::chrono::duration<double,std::milli>(Clock::now()-start).count(); }

template<class T> struct Buffer {
    T* data=nullptr;
    std::size_t size=0;
    explicit Buffer(std::size_t n):size(n) { if(n) check(cudaMalloc(reinterpret_cast<void**>(&data),n*sizeof(T))); }
    explicit Buffer(const std::vector<T>& v):Buffer(v.size()) {
        if(size) check(cudaMemcpy(data,v.data(),size*sizeof(T),cudaMemcpyHostToDevice));
        // Pageable H2D cudaMemcpy may return after staging, before DMA finishes.
        // Tests execute on a nonblocking stream, which does not wait on stream 0.
        check(cudaStreamSynchronize(nullptr));
    }
    ~Buffer() { cudaFree(data); }
    Buffer(const Buffer&)=delete;
    Buffer& operator=(const Buffer&)=delete;
    std::vector<T> host(cudaStream_t stream) const {
        std::vector<T> out(size);
        if(size) check(cudaMemcpyAsync(out.data(),data,size*sizeof(T),cudaMemcpyDeviceToHost,stream));
        check(cudaStreamSynchronize(stream)); return out;
    }
};
struct GPUColumn {
    Buffer<std::uint32_t> windows,slots,offset,swOffset,rows,cacheSlots,cachedRows,counts;
    Buffer<std::uint64_t> bits;
    Buffer<float> values;
    DeviceColumnSWTCF view;
    explicit GPUColumn(const ColumnSWTCF& m):windows(m.colWindowOffset),slots(m.tileRowSlot),offset(m.tileValueOffset),
        swOffset(m.superWindowRowOffset),rows(m.superWindowRows),cacheSlots(m.superWindowRowCacheSlot),
        cachedRows(m.superWindowCachedRowSlot),counts(m.superWindowCachedCount),bits(m.tileLocalBit),values(m.values),
        view{m.rows,m.cols,m.superWindowSize,windows.data,slots.data,offset.data,bits.data,values.data,
            swOffset.data,rows.data,cacheSlots.data,cachedRows.data,counts.data} {}
};
struct GPURow {
    Buffer<std::uint32_t> window,offset,mapping;
    Buffer<std::uint64_t> bits;
    Buffer<float> values;
    std::uint32_t rows,cols;
    explicit GPURow(const RowBitTCF& m):window(m.rowWindowOffset),offset(m.tcOffset),mapping(m.sparseA2B),
        bits(m.tcLocalBit),values(m.values),rows(m.rows),cols(m.cols) {}
    void launch(const float* x,float* y,unsigned f,cudaStream_t stream) const {
        check(launchRowSpMM(window.data,offset.data,mapping.data,bits.data,values.data,x,y,rows,cols,f,stream));
    }
};
inline float tf32(float x) {
    std::uint32_t b; std::memcpy(&b,&x,4); b=(b+4096u)&0xffffe000u; std::memcpy(&x,&b,4); return x;
}
struct Matrix { std::uint32_t rows,cols; CanonicalCSR csr; };
inline Matrix makeMatrix(unsigned rows,unsigned cols,const std::function<float(unsigned,unsigned)>& weight) {
    CanonicalCSR raw;
    for(unsigned r=0;r<rows;++r) {
        for(unsigned c=0;c<cols;++c) {
            float v=weight(r,c); if(v!=0.f) { raw.src.push_back(c); raw.values.push_back(v); }
        }
        raw.ptr.push_back(raw.src.size());
    }
    return {rows,cols,raw};
}
inline std::vector<float> features(unsigned cols,unsigned f) {
    std::vector<float> x(std::size_t(cols)*f);
    std::mt19937 rng(17); std::uniform_real_distribution<float> dist(-1,1);
    for(auto& v:x) v=dist(rng); return x;
}
inline std::vector<double> reference(const Matrix& m,const std::vector<float>& x,unsigned f,bool rounded) {
    std::vector<double> out(std::size_t(m.rows)*f,0);
    for(unsigned r=0;r<m.rows;++r)
        for(auto e=m.csr.ptr[r];e<m.csr.ptr[r+1];++e)
            for(unsigned j=0;j<f;++j) {
                float a=m.csr.values[e],b=x[std::size_t(m.csr.src[e])*f+j];
                out[std::size_t(r)*f+j]+=double(rounded?tf32(a):a)*(rounded?tf32(b):b);
            }
    return out;
}
struct Error { double rmse,maxAbsolute,maxReference; };
template<class T> inline Error error(const std::vector<float>& output,const std::vector<T>& expected) {
    require(output.size()==expected.size(),"reference size mismatch");
    double sum=0,denom=0,maxAbs=0,maxRef=0;
    for(std::size_t i=0;i<output.size();++i) {
        require(std::isfinite(output[i]),"nonfinite output");
        double d=output[i]-double(expected[i]); sum+=d*d; denom+=double(expected[i])*expected[i];
        maxAbs=std::max(maxAbs,std::abs(d)); maxRef=std::max(maxRef,std::abs(double(expected[i])));
    }
    return {std::sqrt(sum/std::max(denom,1e-24)),maxAbs,maxRef};
}
inline void verify(Error e) { require(e.rmse<2e-4 && e.maxAbsolute<2e-4+e.maxReference*5e-4,"TF32 reference mismatch"); }
// Capture complete independent operations, including clear for Column.
inline float measure(const std::function<void()>& operation,int repeat,cudaStream_t stream) {
    for(int i=0;i<10;++i) operation();
    check(cudaStreamSynchronize(stream));
    cudaGraph_t g; cudaGraphExec_t exec;
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    for(int i=0;i<repeat;++i) operation();
    check(cudaStreamEndCapture(stream,&g)); check(cudaGraphInstantiate(&exec,g,nullptr,nullptr,0));
    cudaEvent_t begin,end; check(cudaEventCreate(&begin)); check(cudaEventCreate(&end));
    std::vector<float> times;
    for(int s=0;s<5;++s) {
        check(cudaEventRecord(begin,stream)); check(cudaGraphLaunch(exec,stream));
        check(cudaEventRecord(end,stream)); check(cudaEventSynchronize(end));
        float t; check(cudaEventElapsedTime(&t,begin,end)); times.push_back(t/repeat);
    }
    cudaEventDestroy(begin);cudaEventDestroy(end);cudaGraphExecDestroy(exec);cudaGraphDestroy(g);
    std::sort(times.begin(),times.end()); return times[2];
}
inline void decode(const ColumnSWTCF& m,const CanonicalCSR& csr) {
    std::vector<std::vector<std::pair<unsigned,float>>> entries(m.rows);
    for(std::size_t w=0;w+1<m.colWindowOffset.size();++w) {
        auto sw=w/m.superWindowSize;
        for(auto t=m.colWindowOffset[w];t<m.colWindowOffset[w+1];++t) {
            auto v=m.tileValueOffset[t]; auto bits=m.tileLocalBit[t];
            for(unsigned bit=0;bit<64;++bit) if(bits & (1ULL<<bit)) {
                auto slot=m.tileRowSlot[std::size_t(t)*8+bit/8];
                require(slot!=UINT32_MAX,"nonzero in padded row");
                auto row=m.superWindowRows[m.superWindowRowOffset[sw]+slot];
                entries.at(row).emplace_back(w*8+bit%8,m.values.at(v++));
            }
            require(v==m.tileValueOffset[t+1],"value/bitmap ordering mismatch");
        }
    }
    for(unsigned r=0;r<m.rows;++r) {
        std::sort(entries[r].begin(),entries[r].end());
        require(entries[r].size()==csr.ptr[r+1]-csr.ptr[r],"decoded row length mismatch");
        for(std::size_t i=0;i<entries[r].size();++i) {
            auto e=csr.ptr[r]+i;
            require(entries[r][i].first==csr.src[e] && entries[r][i].second==csr.values[e],"decoded matrix mismatch");
        }
    }
}
}
