#include "matrix_io.cuh"
#include "q4_utils.cuh"
#include "acc_native_reference.cuh"
#include <array>
#include <memory>
#include <numeric>
#include <iomanip>
using namespace test;
__global__ void packFeatures(const float* x,float* packed,unsigned n,unsigned F,unsigned base,unsigned width) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=size_t(n)*width)return;
    unsigned j=i%width;packed[i]=base+j<F?x[(i/width)*F+base+j]:0.f;
}
__global__ void unpackFeatures(const float* packed,float* y,unsigned n,unsigned F,unsigned base,unsigned width) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=size_t(n)*width)return;
    unsigned j=i%width;if(base+j<F)y[(i/width)*F+base+j]=packed[i];
}
__device__ float rna(float x) {unsigned b;asm("cvt.rna.tf32.f32 %0, %1;":"=r"(b):"f"(x));return __uint_as_float(b);}
__global__ void referenceCSRNative(const unsigned* ptr,const unsigned* col,const float* a,const float* x,float* y,unsigned n,unsigned F) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=size_t(n)*F)return;
    unsigned r=i/F,j=i%F;double sum=0;
    for(unsigned e=ptr[r];e<ptr[r+1];++e)sum+=double(rna(a[e]))*rna(x[size_t(col[e])*F+j]);
    y[i]=float(sum);
}
struct Chunk {
    unsigned base,width;
    Buffer<float> x,y;
    Chunk(unsigned b,unsigned w,unsigned n,unsigned rows):base(b),width(w),x(size_t(n)*w),y(size_t((rows+7)/8*8)*w){}
};
struct Method {
    std::string name;
    std::function<void()> operation;
    Error error{};
    cudaGraph_t graph{};cudaGraphExec_t exec{};
    std::array<float,15> samples{};
};
void run(const Matrix& input,const std::vector<float>& x,unsigned F,const std::string& prefix,bool smallTest) {
    require(F>0&&F<65536,"invalid native feature dimension");
    cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    Buffer<float> dx(x),dy(size_t(input.rows)*F),referenceY(size_t(input.rows)*F);
    Buffer<unsigned> ptr(input.csr.ptr),cols(input.csr.src);Buffer<float> weights(input.csr.values);
    referenceCSRNative<<<(referenceY.size+255)/256,256,0,stream>>>(ptr.data,cols.data,weights.data,dx.data,referenceY.data,input.rows,F);
    check(cudaGetLastError());auto expected=referenceY.host(stream);
    if(smallTest)verify(error(expected,reference(input,x,F,true)));
    auto t=Clock::now();auto acc=convertAccRow(input.rows,input.cols,input.csr);double accMs=elapsed(t);GPURow da(acc);
    auto grouped=buildQ4(input.rows,input.cols,input.csr,true);validateQ4Cache(grouped,32);GPUQ4 gg(grouped);
    std::vector<std::unique_ptr<Chunk>> chunks;
    for(unsigned base=0;base<F;base+=512)chunks.emplace_back(new Chunk(base,std::min(512u,((F-base+127)/128)*128),input.cols,input.rows));
    auto pack=[&]{for(auto& c:chunks){packFeatures<<<(c->x.size+255)/256,256,0,stream>>>(dx.data,c->x.data,input.cols,F,c->base,c->width);check(cudaGetLastError());}};
    auto unpack=[&]{for(auto& c:chunks){unpackFeatures<<<(size_t(input.rows)*c->width+255)/256,256,0,stream>>>(c->y.data,dy.data,input.rows,F,c->base,c->width);check(cudaGetLastError());}};
    auto compute=[&](unsigned method){for(auto& c:chunks){
        check(cudaMemsetAsync(c->y.data,0,c->y.size*4,stream));
        if(method==0)check(launchAccRowSpMMBarrier(da.window.data,da.offset.data,da.mapping.data,da.bits.data,da.values.data,c->x.data,c->y.data,(input.rows+7)/8*8,input.cols,c->width,stream));
        else check(launchQ4(gg.view,c->x.data,c->y.data,c->width,Q4Config{method==1?Q4Version::Float4:Q4Version::Pipeline,32,false},stream,false));
    }};
    pack();check(cudaStreamSynchronize(stream));
    // Exact verification of the adapter's padding and source-column mapping.
    if(smallTest)for(auto& c:chunks){auto h=c->x.host(stream);for(unsigned r=0;r<input.cols;++r)for(unsigned j=0;j<c->width;++j)
        require(h[size_t(r)*c->width+j]==(c->base+j<F?x[size_t(r)*F+c->base+j]:0.f),"feature adapter mismatch");}
    std::vector<Method> methods;
    const char* names[]={"Acc_barrier","Q4_float4_H32","Q4_float4_pipeline_H32"};
    for(unsigned k=0;k<3;++k)for(bool online:{false,true}) {
        Method m;m.name=std::string(names[k])+(online?"_native":"_packed");
        m.operation=[&,k,online]{if(online)pack();compute(k);if(online)unpack();};
        check(cudaMemsetAsync(dy.data,0xff,dy.size*4,stream));m.operation();if(!online)unpack();
        m.error=error(dy.host(stream),expected);verify(m.error);
        // Padding output must stay zero, including every final tail feature.
        if(smallTest)for(auto& c:chunks){auto h=c->y.host(stream);for(unsigned r=0;r<input.rows;++r)for(unsigned j=0;j<c->width;++j)
            if(c->base+j>=F)require(h[size_t(r)*c->width+j]==0.f,"nonzero padding output");}
        methods.push_back(std::move(m));
    }
    if(smallTest) {std::cout<<"native F="<<F<<" all methods and adapters PASS\n";check(cudaStreamDestroy(stream));return;}
    const int repeats=input.rows<50000?100:10;
    for(auto& m:methods){
        check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));for(int i=0;i<repeats;++i)m.operation();
        check(cudaStreamEndCapture(stream,&m.graph));check(cudaGraphInstantiate(&m.exec,m.graph,nullptr,nullptr,0));
        check(cudaGraphLaunch(m.exec,stream));
    }
    check(cudaStreamSynchronize(stream));
    cudaEvent_t begin,end;check(cudaEventCreate(&begin));check(cudaEventCreate(&end));
    std::ofstream order(prefix+"_order.csv");require(bool(order),"cannot write order CSV");order<<"round,position,method,ms\n"<<std::setprecision(9);
    std::vector<unsigned> sequence(methods.size());std::iota(sequence.begin(),sequence.end(),0);std::mt19937 rng(20261019);
    for(unsigned round=0;round<15;++round){std::shuffle(sequence.begin(),sequence.end(),rng);
        for(unsigned pos=0;pos<sequence.size();++pos){auto& m=methods[sequence[pos]];
            check(cudaEventRecord(begin,stream));check(cudaGraphLaunch(m.exec,stream));check(cudaEventRecord(end,stream));check(cudaEventSynchronize(end));
            float ms;check(cudaEventElapsedTime(&ms,begin,end));m.samples[round]=ms/repeats;
            order<<round<<','<<pos<<','<<m.name<<','<<m.samples[round]<<'\n'<<std::flush;
        }
        std::cerr<<"completed round "<<round<<'\n';
    }
    std::ofstream csv(prefix+".csv");require(bool(csv),"cannot write results CSV");csv<<std::setprecision(9);
    csv<<"method,rows,nnz,F,padded_F,chunks,repeats,median_ms,min_ms,max_ms,relative_l2,max_abs,acc_format_ms,q4_group_ms,q4_format_ms,q4_plan_ms,T,P,U,peak";
    for(unsigned i=0;i<15;++i)csv<<",sample"<<i<<"_ms";csv<<'\n';
    for(auto& m:methods){auto sorted=m.samples;std::sort(sorted.begin(),sorted.end());
        csv<<m.name<<','<<input.rows<<','<<input.csr.src.size()<<','<<F<<','<<(F+127)/128*128<<','<<chunks.size()<<','<<repeats<<','<<sorted[7]<<','<<sorted.front()<<','<<sorted.back()<<','<<m.error.rmse<<','<<m.error.maxAbsolute<<','<<accMs<<','<<grouped.groupingMs<<','<<grouped.formatMs<<','<<grouped.planMs<<','<<grouped.bitmap.size()<<','<<grouped.P<<','<<grouped.U<<','<<grouped.peakSlots;
        for(auto v:m.samples)csv<<','<<v;csv<<'\n';check(cudaGraphExecDestroy(m.exec));check(cudaGraphDestroy(m.graph));
    }
    check(cudaEventDestroy(begin));check(cudaEventDestroy(end));check(cudaStreamDestroy(stream));
}
int main(int argc,char** argv){try{
    if(argc==2&&std::string(argv[1])=="--test"){
        auto m=makeMatrix(19,23,[](unsigned r,unsigned c){return (r+c)%3==0?float(int(c%5)-2)*.25f:0.f;});
        for(unsigned f:{1,100,128,384,500})run(m,features(m.cols,f),f,"",true);
    }else{
        require(argc==5,"usage: benchmark_native_features matrix.csr nativeF X.bin output_prefix | --test");
        auto m=readMatrix(argv[1]);unsigned f=std::stoul(argv[2]);std::ifstream file(argv[3],std::ios::binary|std::ios::ate);
        require(bool(file)&&std::uint64_t(file.tellg())==std::uint64_t(m.cols)*f*4,"X file must match original feature dimension exactly");
        auto x=readFeatures(argv[3],m.cols,f);run(m,x,f,argv[4],false);
    }
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
