#include "matrix_io.cuh"
#include "q4_utils.cuh"
#include "acc_reference.cuh"
#include <iomanip>
using namespace test;
int main(int argc,char** argv) {
    try {
        if(argc<3){std::cout<<"benchmark_q4 matrix.csr F [device=0] [X.bin]\n";return 0;}
        check(cudaSetDevice(argc>3?std::stoi(argv[3]):0));unsigned F=std::stoul(argv[2]);require(F==128||F==256,"F must be 128/256");
        cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        auto input=readMatrix(argv[1]);auto x=readFeatures(argc>4?argv[4]:nullptr,input.cols,F);
        unsigned padded=(input.rows+7)/8*8;std::size_t real=std::size_t(input.rows)*F;
        Buffer<float> dx(x),dy(std::size_t(padded)*F);
        auto start=Clock::now();auto acc=convertAccRow(input.rows,input.cols,input.csr);double accMs=elapsed(start);
        GPURow da(acc);
        auto launchAcc=[&]{check(cudaMemsetAsync(dy.data,0,dy.size*4,stream));check(launchAccRowSpMM(da.window.data,da.offset.data,da.mapping.data,da.bits.data,da.values.data,dx.data,dy.data,padded,input.cols,F,stream));};
        launchAcc();auto expected=dy.host(stream);expected.resize(real);
        // Independent sampled CSR reference also checks Acc mapping against original IDs.
        for(unsigned i=0;i<64;++i) {
            unsigned r=std::uint64_t(i)*(input.rows-1)/63;
            for(unsigned f=0;f<F;f+=17) {
                double ref=0;for(auto e=input.csr.ptr[r];e<input.csr.ptr[r+1];++e)ref+=double(tf32(input.csr.values[e]))*tf32(x[std::size_t(input.csr.src[e])*F+f]);
                require(std::abs(expected[std::size_t(r)*F+f]-ref)<.002+.001*std::abs(ref),"Acc vs sampled CSR mismatch");
            }
        }
        std::cout<<"method,F,T,P,U,global_updates,shared_elements,group_ms,format_ms,plan_ms,fallback,peak_slots,complete_ms,registers,shared_bytes,local_bytes,blocks_per_sm,rmse,max_abs\n"<<std::setprecision(9);
        int ar,as,al,ab;check(accRowAttributes(F,&ar,&as,&al,&ab));
        std::cout<<"Acc,"<<F<<','<<acc.tcLocalBit.size()<<",0,0,0,0,0,"<<accMs<<",0,0,0,"<<measure(launchAcc,10,stream)<<','<<ar<<','<<as<<','<<al<<','<<ab<<",0,0\n"<<std::flush;
        for(bool grouped:{false,true}) {
            auto m=buildQ4(input.rows,input.cols,input.csr,grouped);GPUQ4 gpu(m);
            std::vector<Q4Version> versions{Q4Version::Q1};if(grouped)versions={Q4Version::Q1,Q4Version::Sync,Q4Version::Bits,Q4Version::Float4,Q4Version::Pipeline};
            for(auto v:versions) {
                auto launch=[&]{check(launchQ4(gpu.view,dx.data,dy.data,F,v,stream));};
                launch();auto out=dy.host(stream);out.resize(real);auto err=error(out,expected);verify(err);
                auto time=measure(launch,10,stream);ColumnKernelInfo info;check(q4Attributes(v,&info));
                const char* name=v==Q4Version::Q1?(grouped?"Q1_grouped":"Q1_identity"):v==Q4Version::Sync?"Q4_sync":v==Q4Version::Bits?"Q4_bits":v==Q4Version::Float4?"Q4_float4":"Q4_pipeline";
                std::cout<<name<<','<<F<<','<<m.bitmap.size()<<','<<m.P<<','<<m.U<<','<<std::uint64_t(F)*(v==Q4Version::Q1?m.P:m.U)<<','<<(v==Q4Version::Q1?0:2ULL*F*(m.P-m.U))<<','<<m.groupingMs<<','<<m.formatMs<<','<<m.planMs<<','<<m.fallbackGroups<<','<<m.peakSlots<<','<<time<<','<<info.registers<<','<<info.sharedBytes<<','<<info.localBytes<<','<<info.activeBlocksPerSM<<','<<err.rmse<<','<<err.maxAbsolute<<'\n'<<std::flush;
            }
        }
        check(cudaStreamDestroy(stream));
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
