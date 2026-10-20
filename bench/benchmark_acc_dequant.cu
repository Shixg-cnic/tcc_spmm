// Original Acc vs per-TC-block INT4 X loads. No changes to the sparse path.
#include "matrix_io.cuh"
#include "acc_reference.cuh"
#include "int4_format.cuh"
#include <array>
#include <iomanip>
#include <numeric>
using namespace test;
using U=std::uint32_t;
cudaError_t launchAccRowInt4SpMM(const U*,const U*,const U*,const std::uint64_t*,const float*,const std::uint8_t*,const float*,float*,U,U,U,cudaStream_t);

// Independent sampled CSR reference; all F columns, 256 evenly spaced rows.
struct Validation { double rel=0,max=0; std::size_t bad=0; };
Validation validate(const Matrix& m,const std::vector<float>& x,const std::vector<float>& y,U F) {
    double se=0,sr=0;Validation v;
    U count=std::min(256u,m.rows);
    for(U i=0;i<count;++i) {
        U r=std::uint64_t(i)*(m.rows-1)/std::max(1u,count-1);
        for(U f=0;f<F;++f) {
            double ref=0;
            for(U e=m.csr.ptr[r];e<m.csr.ptr[r+1];++e)
                ref+=double(tf32(m.csr.values[e]))*tf32(x[std::size_t(m.csr.src[e])*F+f]);
            double value=y[std::size_t(r)*F+f],d=std::abs(value-ref);
            if(!std::isfinite(value)||d>0.002+0.001*std::abs(ref))++v.bad;
            v.max=std::max(v.max,d);se+=d*d;sr+=ref*ref;
        }
    }
    v.rel=std::sqrt(se/std::max(sr,1e-24));return v;
}
// Full-array comparison against original Acc operating on exactly X-hat.
__global__ void compare(const float* a,const float* b,std::size_t n,unsigned long long* bad) {
    unsigned count=0;
    for(std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=std::size_t(gridDim.x)*blockDim.x)
        count+=!isfinite(a[i])||!isfinite(b[i])||fabsf(a[i]-b[i])>0.002f+0.001f*fabsf(b[i]);
    if(count)atomicAdd(bad,(unsigned long long)count);
}
int main(int argc,char** argv) {try {
    require(argc==6,"usage: benchmark_acc_dequant matrix.csr X.bin F device output_prefix");
    U F=std::stoul(argv[3]);require(F==128||F==256||F==512,"F must be 128/256/512");
    check(cudaSetDevice(std::stoi(argv[4])));cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    auto m=readMatrix(argv[1]);auto x=readFeatures(argv[2],m.cols,F);
    auto acc=convertAccRow(m.rows,m.cols,m.csr);
    for(std::size_t i=1;i<acc.rowWindowOffset.size();++i)
        require(acc.rowWindowOffset[i]>acc.rowWindowOffset[i-1],"original Acc requires nonempty row windows");
    GPURow da(acc);U padded=(m.rows+7)/8*8;std::size_t n=std::size_t(m.rows)*F;
    Buffer<float> dx(x),xh(n),scale(m.cols),y(std::size_t(padded)*F),referenceY(std::size_t(padded)*F);
    Buffer<std::uint8_t> q(std::size_t(m.cols)*F/2);
    check(quantizeXInt4(dx.data,q.data,scale.data,m.cols,F,stream));
    check(dequantizeXInt4(q.data,scale.data,xh.data,m.cols,F,stream));
    auto xhat=xh.host(stream);
    // Verify unpack/scale against an independent CPU quantizer, not just GPU round trip.
    auto packed=q.host(stream);auto scales=scale.host(stream);
    for(U r=0;r<m.cols;++r) {
        float mx=0;for(U f=0;f<F;++f)mx=std::max(mx,std::abs(x[std::size_t(r)*F+f]));
        float s=mx==0?1:mx/7;require(scales[r]==s,"scale mismatch");
        for(U f=0;f<F;++f) {
            auto i=std::size_t(r)*F+f;int v=(packed[i/2]>>((f&1)*4))&15;if(v&8)v-=16;
            int expected=std::max(-7,std::min(7,int(std::nearbyint(x[i]/s))));
            require(v==expected && xhat[i]==float(v)*s,"INT4 quantization mismatch");
        }
    }
    packed.clear();packed.shrink_to_fit();
    std::array<std::string,3> names{"Acc_original_X","Acc_original_Xhat","Acc_INT4_fused"};
    auto launch=[&](unsigned method,float* out){
        if(method==2)check(launchAccRowInt4SpMM(da.window.data,da.offset.data,da.mapping.data,da.bits.data,da.values.data,q.data,scale.data,out,padded,m.cols,F,stream));
        else check(launchAccRowSpMM(da.window.data,da.offset.data,da.mapping.data,da.bits.data,da.values.data,method==0?dx.data:xh.data,out,padded,m.cols,F,stream));
    };
    std::string prefix=argv[5];std::ofstream validation(prefix+"_validation.csv");
    validation<<"method,pass,sampled_relative_l2,sampled_max_abs,sampled_bad,full_vs_xhat_bad\n"<<std::setprecision(10);
    Buffer<unsigned long long> bad(1);
    for(unsigned pass=0;pass<3;++pass) {
        check(cudaMemsetAsync(referenceY.data,0xff,referenceY.size*4,stream));launch(1,referenceY.data);
        for(unsigned method=0;method<3;++method) {
            check(cudaMemsetAsync(y.data,0xff,y.size*4,stream));launch(method,y.data);auto host=y.host(stream);
            auto v=validate(m,method==0?x:xhat,host,F);
            unsigned long long count=0;
            if(method!=0) {
                check(cudaMemsetAsync(bad.data,0,8,stream));compare<<<4096,256,0,stream>>>(y.data,referenceY.data,n,bad.data);check(cudaGetLastError());count=bad.host(stream)[0];
            }
            validation<<names[method]<<','<<pass<<','<<v.rel<<','<<v.max<<','<<v.bad<<','<<count<<'\n'<<std::flush;
            std::cerr<<names[method]<<" pass="<<pass<<" sampled_bad="<<v.bad<<" full_bad="<<count<<'\n';
        }
    }
    // Quantization changes the mathematical input. Report its sampled output error separately.
    launch(2,y.data);auto quantError=validate(m,x,y.host(stream),F);
    std::ofstream meta(prefix+"_meta.csv");
    meta<<"nodes,nnz,F,tc_blocks,quant_output_relative_l2,quant_output_max_abs\n"<<std::setprecision(10)
        <<m.rows<<','<<m.csr.src.size()<<','<<F<<','<<acc.tcLocalBit.size()<<','<<quantError.rel<<','<<quantError.max<<'\n';
    std::array<cudaGraph_t,3> graph;std::array<cudaGraphExec_t,3> exec;
    constexpr int repeats=10,rounds=15;
    for(unsigned method=0;method<3;++method) {
        check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        for(int i=0;i<repeats;++i)launch(method,y.data);
        check(cudaStreamEndCapture(stream,&graph[method]));check(cudaGraphInstantiate(&exec[method],graph[method],nullptr,nullptr,0));
        for(int i=0;i<3;++i)check(cudaGraphLaunch(exec[method],stream));check(cudaStreamSynchronize(stream));
    }
    cudaEvent_t start,end;check(cudaEventCreate(&start));check(cudaEventCreate(&end));
    std::mt19937 rng(20261020);std::array<unsigned,3> order{0,1,2};
    std::ofstream samples(prefix+"_samples.csv");samples<<"round,position,method,kernel_ms\n"<<std::setprecision(10);
    for(int round=0;round<rounds;++round) {
        std::shuffle(order.begin(),order.end(),rng);
        for(unsigned pos=0;pos<3;++pos) {
            unsigned method=order[pos];check(cudaEventRecord(start,stream));check(cudaGraphLaunch(exec[method],stream));check(cudaEventRecord(end,stream));check(cudaEventSynchronize(end));
            float ms;check(cudaEventElapsedTime(&ms,start,end));samples<<round<<','<<pos<<','<<names[method]<<','<<ms/repeats<<'\n'<<std::flush;
        }
        std::cerr<<"timing round "<<round<<" complete\n";
    }
    for(unsigned i=0;i<3;++i){check(cudaGraphExecDestroy(exec[i]));check(cudaGraphDestroy(graph[i]));}
    check(cudaEventDestroy(start));check(cudaEventDestroy(end));check(cudaStreamDestroy(stream));
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
