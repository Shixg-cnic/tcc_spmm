#include "matrix_io.cuh"
#include "q4_utils.cuh"
#include "acc_reference.cuh"
#include <array>
#include <iomanip>
#include <numeric>
using namespace test;
namespace {
struct Method {
    std::string name;
    std::function<void()> launch;
    const Q4Format* format=nullptr;
    Q4Config config{};
    unsigned h=0;
    bool shuffled=false;
    ColumnKernelInfo info{};
    Error err{};
    cudaGraph_t graph=nullptr;
    cudaGraphExec_t exec=nullptr;
    std::array<float,15> samples{};
    std::array<unsigned,15> positions{};
};
void structure(const Q4Format& m,const std::vector<Q4Task>& shuffled,const std::string& prefix) {
    std::ofstream sw(prefix+"_sw.csv"),cw(prefix+"_cw.csv"),quant(prefix+"_quantiles.csv");
    require(bool(sw)&&bool(cw)&&bool(quant),"cannot write structure files");
    sw<<"task,tiles,P,U,peak_slots,shuffled_position\n";cw<<"cw,tiles\n";
    std::vector<unsigned> position(m.tasks.size());
    for(unsigned i=0;i<shuffled.size();++i)position.at(shuffled[i].cw[0]/4)=i;
    std::vector<std::uint64_t> lengths,cwLengths,ps,us,peaks;
    for(unsigned i=0;i<m.tasks.size();++i) {
        auto t=m.tasks[i];auto s=m.taskStats.at(i);unsigned n=t.end-t.begin;
        sw<<i<<','<<n<<','<<s.P<<','<<s.U<<','<<s.peakSlots<<','<<position[i]<<'\n';
        lengths.push_back(n);ps.push_back(s.P);us.push_back(s.U);peaks.push_back(s.peakSlots);
    }
    for(unsigned i=0;i+1<m.cwOffset.size();++i) {unsigned n=m.cwOffset[i+1]-m.cwOffset[i];cw<<i<<','<<n<<'\n';cwLengths.push_back(n);}
    quant<<"metric,p0,p25,p50,p75,p90,p95,p99,max\n";
    auto emit=[&](const char* name,std::vector<std::uint64_t>& v){
        std::sort(v.begin(),v.end());quant<<name;
        for(unsigned p:{0,25,50,75,90,95,99,100})quant<<','<<(v.empty()?0:v[std::size_t(p)*(v.size()-1)/100]);quant<<'\n';
    };
    emit("cw_tiles",cwLengths);emit("sw_tiles",lengths);emit("sw_P",ps);emit("sw_U",us);emit("sw_peak",peaks);
}
}
int main(int argc,char** argv) {
    try {
        if(argc<6){std::cout<<"benchmark_q4_datasets matrix.csr F device X.bin-or-minus output_prefix\n";return 0;}
        check(cudaSetDevice(std::stoi(argv[3])));unsigned F=std::stoul(argv[2]);require(F==128||F==256,"F must be 128/256");
        std::string prefix=argv[5];std::ofstream csv(prefix+".csv"),ordering(prefix+"_measurement_order.csv");
        require(bool(csv)&&bool(ordering),"cannot write output");
        cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        auto input=readMatrix(argv[1]);auto x=readFeatures(std::string(argv[4])=="-"?nullptr:argv[4],input.cols,F);
        unsigned padded=(input.rows+7)/8*8;std::size_t real=std::size_t(input.rows)*F;
        Buffer<float> dx(x),dy(std::size_t(padded)*F);
        auto start=Clock::now();auto acc=convertAccRow(input.rows,input.cols,input.csr);double accMs=elapsed(start);GPURow da(acc);
        auto launchAcc=[&]{check(cudaMemsetAsync(dy.data,0,dy.size*4,stream));check(launchAccRowSpMM(da.window.data,da.offset.data,da.mapping.data,da.bits.data,da.values.data,dx.data,dy.data,padded,input.cols,F,stream));};
        launchAcc();auto expected=dy.host(stream);expected.resize(real);
        for(unsigned i=0;i<64;++i) {
            unsigned r=std::uint64_t(i)*(input.rows-1)/63;
            for(unsigned f=0;f<F;f+=17) {
                double ref=0;for(auto e=input.csr.ptr[r];e<input.csr.ptr[r+1];++e)ref+=double(tf32(input.csr.values[e]))*tf32(x[std::size_t(input.csr.src[e])*F+f]);
                require(std::abs(expected[std::size_t(r)*F+f]-ref)<.002+.001*std::abs(ref),"Acc vs sampled CSR mismatch");
            }
        }
        std::cerr<<"building grouped format\n";
        auto grouped=buildQ4(input.rows,input.cols,input.csr,true);
        auto shuffled=reorderQ4Tasks(grouped.tasks);
        structure(grouped,shuffled,prefix);
        GPUQ4 gg(grouped);
        std::vector<Method> methods;
        Method baseline;baseline.name="Acc";baseline.launch=launchAcc;
        check(accRowAttributes(F,&baseline.info.registers,&baseline.info.sharedBytes,&baseline.info.localBytes,&baseline.info.activeBlocksPerSM));methods.push_back(baseline);
        auto add=[&](std::string name,const Q4Format& format,DeviceQ4 view,Q4Config config,bool shuffled=false){
            Method method;method.name=name;method.format=&format;method.config=config;method.shuffled=shuffled;
            method.h=config.version==Q4Version::Q1?0:q4CacheSlots(format.peakSlots,config.cacheSlots);
            if(method.h)validateQ4Cache(format,method.h);
            check(q4Attributes(config,format.peakSlots,&method.info));
            method.launch=[&,view,config]{check(launchQ4(view,dx.data,dy.data,F,config,stream));};methods.push_back(method);
        };
        add("Q4_float4_H32",grouped,gg.view,{Q4Version::Float4,32,false});
        add("Q4_float4_pipeline_H32",grouped,gg.view,{Q4Version::Pipeline,32,false});
        const int repeats=input.rows<50000?1000:10;
        for(auto& m:methods) {
            m.launch();auto out=dy.host(stream);out.resize(real);m.err=error(out,expected);verify(m.err);
            check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
            for(int i=0;i<repeats;++i)m.launch();
            check(cudaStreamEndCapture(stream,&m.graph));check(cudaGraphInstantiate(&m.exec,m.graph,nullptr,nullptr,0));
            std::cerr<<"verified "<<m.name<<" H="<<m.h<<" regs="<<m.info.registers<<" shared="<<m.info.sharedBytes<<" blocks="<<m.info.activeBlocksPerSM<<'\n';
        }
        // Complete independent calls including output clear; more repeats for small graphs.
        // Warm every candidate before timing. Interleave in a seeded order per round.
        for(auto& m:methods)check(cudaGraphLaunch(m.exec,stream));check(cudaStreamSynchronize(stream));
        cudaEvent_t begin,end;check(cudaEventCreate(&begin));check(cudaEventCreate(&end));
        std::mt19937 rng(20261018);std::vector<unsigned> order(methods.size());std::iota(order.begin(),order.end(),0);
        ordering<<"round,position,method,complete_ms\n"<<std::setprecision(9);
        for(unsigned round=0;round<15;++round) {
            std::shuffle(order.begin(),order.end(),rng);
            for(unsigned pos=0;pos<order.size();++pos) {
                auto& m=methods[order[pos]];
                check(cudaEventRecord(begin,stream));check(cudaGraphLaunch(m.exec,stream));
                check(cudaEventRecord(end,stream));check(cudaEventSynchronize(end));
                float elapsed;check(cudaEventElapsedTime(&elapsed,begin,end));m.samples[round]=elapsed/repeats;m.positions[round]=pos;
                ordering<<round<<','<<pos<<','<<m.name<<','<<m.samples[round]<<'\n'<<std::flush;
            }
            std::cerr<<"completed interleaved round "<<round<<'\n';
        }
        csv<<"method,F,H,xor_A,shuffled,T,P,U,global_updates,shared_elements,group_ms,format_ms,plan_ms,fallback,peak_slots,complete_ms,min_ms,max_ms,range_pct";
        for(unsigned i=0;i<15;++i)csv<<",sample"<<i<<"_ms";
        csv<<",registers,shared_bytes,local_bytes,blocks_per_sm,rmse,max_abs,repeats\n"<<std::setprecision(9);
        for(auto& m:methods) {
            auto samples=m.samples;std::sort(samples.begin(),samples.end());auto f=m.format;
            bool q1=m.config.version==Q4Version::Q1;
            csv<<m.name<<','<<F<<','<<m.h<<','<<m.config.xorA<<','<<m.shuffled<<','<<(f?f->bitmap.size():acc.tcLocalBit.size())<<','<<(f?f->P:0)<<','<<(f?f->U:0)<<','<<(f?std::uint64_t(F)*(q1?f->P:f->U):0)<<','<<(f&&!q1?2ULL*F*(f->P-f->U):0)<<','<<(f?f->groupingMs:0)<<','<<(f?f->formatMs:accMs)<<','<<(f?f->planMs:0)<<','<<(f?f->fallbackGroups:0)<<','<<(f?f->peakSlots:0)<<','<<samples[7]<<','<<samples.front()<<','<<samples.back()<<','<<100*(samples.back()-samples.front())/samples[7];
            for(auto s:m.samples)csv<<','<<s;
            csv<<','<<m.info.registers<<','<<m.info.sharedBytes<<','<<m.info.localBytes<<','<<m.info.activeBlocksPerSM<<','<<m.err.rmse<<','<<m.err.maxAbsolute<<','<<repeats<<'\n';
            check(cudaGraphExecDestroy(m.exec));check(cudaGraphDestroy(m.graph));
        }
        check(cudaEventDestroy(begin));check(cudaEventDestroy(end));check(cudaStreamDestroy(stream));
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
