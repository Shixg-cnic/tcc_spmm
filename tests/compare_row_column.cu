#include "matrix_io.cuh"
#include <iomanip>
#include <memory>
using namespace test;

struct Structure {
    double density,mean,p95,maximum,swRowsMean,swRowsMax,repeatedMean,cacheUsage;
    std::uint64_t uncached,cached,saved;
};
Structure structure(const ColumnSWTCF& m) {
    std::vector<std::uint32_t> tiles;
    for(std::size_t w=1;w<m.colWindowOffset.size();++w) tiles.push_back(m.colWindowOffset[w]-m.colWindowOffset[w-1]);
    std::sort(tiles.begin(),tiles.end());
    std::uint64_t occurrences=0,saved=0,totalRows=0,repeated=0,cacheCount=0,maxRows=0;
    for(std::size_t s=0;s<m.superWindowCachedCount.size();++s) {
        auto count=m.superWindowRowOffset[s+1]-m.superWindowRowOffset[s];
        totalRows+=count;maxRows=std::max<std::uint64_t>(maxRows,count);cacheCount+=m.superWindowCachedCount[s];
        std::vector<unsigned> freq(count,0);
        auto end=std::min<std::size_t>((s+1)*m.superWindowSize,tiles.size());
        for(std::size_t w=s*m.superWindowSize;w<end;++w)
            for(auto t=m.colWindowOffset[w];t<m.colWindowOffset[w+1];++t)
                for(unsigned r=0;r<8;++r) {
                    auto slot=m.tileRowSlot[std::size_t(t)*8+r];
                    if(slot!=UINT32_MAX) {++freq[slot];++occurrences;}
                }
        for(unsigned r=0;r<count;++r) {
            repeated+=freq[r]>1;
            if(m.superWindowRowCacheSlot[m.superWindowRowOffset[s]+r]!=UINT32_MAX) saved+=freq[r]-1;
        }
    }
    auto supers=m.superWindowCachedCount.size();
    return {m.tileLocalBit.empty()?0.:double(m.values.size())/(64*m.tileLocalBit.size()),
        double(m.tileLocalBit.size())/tiles.size(),double(tiles[(tiles.size()-1)*95/100]),double(tiles.back()),
        double(totalRows)/supers,double(maxRows),double(repeated)/supers,double(cacheCount)/(supers*COLUMN_CACHE_SIZE),
        occurrences,occurrences-saved,saved};
}
std::size_t bytes(const ColumnSWTCF& m) {
    return 4*(m.colWindowOffset.size()+m.tileRowSlot.size()+m.tileValueOffset.size()+m.values.size()+
        m.superWindowRowOffset.size()+m.superWindowRows.size()+m.superWindowRowCacheSlot.size()+
        m.superWindowCachedRowSlot.size()+m.superWindowCachedCount.size())+8*m.tileLocalBit.size();
}
std::size_t bytes(const RowBitTCF& m) {
    return 4*(m.rowWindowOffset.size()+m.tcOffset.size()+m.sparseA2B.size()+m.values.size())+8*m.tcLocalBit.size();
}

// Isolate kernel Event intervals while every graph operation still clears Y.
// The clear is outside each begin/end interval; timing events remain in the graph.
float kernelTime(const std::function<void()>& clear,const std::function<void()>& kernel,int repeat,cudaStream_t stream) {
    std::vector<cudaEvent_t> start(repeat),stop(repeat);
    for(int i=0;i<repeat;++i) {check(cudaEventCreate(&start[i]));check(cudaEventCreate(&stop[i]));}
    cudaGraph_t g;cudaGraphExec_t exec;
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    for(int i=0;i<repeat;++i) {
        clear();
        check(cudaEventRecordWithFlags(start[i],stream,cudaEventRecordExternal));
        kernel();
        check(cudaEventRecordWithFlags(stop[i],stream,cudaEventRecordExternal));
    }
    check(cudaStreamEndCapture(stream,&g));check(cudaGraphInstantiate(&exec,g,nullptr,nullptr,0));
    std::vector<float> samples;
    for(int s=0;s<5;++s) {
        check(cudaGraphLaunch(exec,stream));check(cudaStreamSynchronize(stream));float sum=0;
        for(int i=0;i<repeat;++i) {float t;check(cudaEventElapsedTime(&t,start[i],stop[i]));sum+=t;}
        samples.push_back(sum/repeat);
    }
    for(int i=0;i<repeat;++i) {cudaEventDestroy(start[i]);cudaEventDestroy(stop[i]);}
    cudaGraphExecDestroy(exec);cudaGraphDestroy(g);std::sort(samples.begin(),samples.end());return samples[2];
}

Error sampledCPU(const Matrix& m,const std::vector<float>& x,const std::vector<float>& y,unsigned f,bool rounded) {
    std::vector<double> ref;std::vector<float> sampled;
    const unsigned count=std::min(64u,m.rows);
    for(unsigned i=0;i<count;++i) {
        auto row=std::uint64_t(i)*(m.rows-1)/std::max(1u,count-1);
        for(unsigned j=0;j<f;++j) {
            double sum=0;
            for(auto e=m.csr.ptr[row];e<m.csr.ptr[row+1];++e) {
                float a=m.csr.values[e],b=x[std::size_t(m.csr.src[e])*f+j];
                sum+=double(rounded?tf32(a):a)*(rounded?tf32(b):b);
            }
            ref.push_back(sum);sampled.push_back(y[row*f+j]);
        }
    }
    return error(sampled,ref);
}
int main(int argc,char** argv) {
    try {
        if(argc<2 || std::string(argv[1])=="--help") {
            std::cout<<"Usage: compare_row_column matrix.{mtx,csr} [F=all|32] [repeat=30] [device=0] [X.bin]\n"
                <<"all: 32,64,128,256,512. X.bin is K*sourceF raw FP32; truncate/cyclic-expand features.\n";return 0;
        }
        require(argc<=6,"too many arguments");
        int repeat=argc>3?std::stoi(argv[3]):30;require(repeat>0,"repeat must be positive");
        check(cudaSetDevice(argc>4?std::stoi(argv[4]):0));cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        auto m=readMatrix(argv[1]);auto start=Clock::now();auto row=convertRow(m.rows,m.cols,m.csr);double rowConvert=elapsed(start);
        start=Clock::now();auto col=convertColumn(m.rows,m.cols,m.csr);double colConvert=elapsed(start);
        // Decode is independent from MMA; use on every matrix, including real inputs.
        decode(col,m.csr);auto st=structure(col);
        check(cudaDeviceSynchronize());start=Clock::now();GPURow gr(row);check(cudaDeviceSynchronize());double rowUpload=elapsed(start);
        start=Clock::now();GPUColumn gc(col);check(cudaDeviceSynchronize());double colUpload=elapsed(start);
        std::cout<<std::setprecision(9)<<"# matrix="<<argv[1]<<" M="<<m.rows<<" K="<<m.cols<<" nnz="<<m.csr.src.size()
            <<" sw="<<col.superWindowSize<<" cache="<<COLUMN_CACHE_SIZE<<" feature_tile="<<COLUMN_FEATURE_TILE<<" cpu_reference_rows="<<std::min(64u,m.rows)<<'\n';
        std::cout<<"# column_tiles="<<col.tileLocalBit.size()<<" density="<<st.density<<" cw_tiles_mean="<<st.mean
            <<" cw_tiles_p95="<<st.p95<<" cw_tiles_max="<<st.maximum<<" sw_rows_mean="<<st.swRowsMean<<" sw_rows_max="<<st.swRowsMax
            <<" sw_repeated_rows_mean="<<st.repeatedMean<<" cache_usage="<<st.cacheUsage<<" row_bytes="<<bytes(row)<<" column_bytes="<<bytes(col)<<'\n';
        std::cout<<"F,method,convert_cpu_ms,format_upload_wall_ms,kernel_event_ms,clear_graph_ms,spmm_graph_ms,relative_rmse_row,max_abs_row,relative_rmse_tf32_sample,max_abs_tf32_sample,relative_rmse_fp32_sample,max_abs_fp32_sample,repeat_max_abs,global_atomic_updates,atomic_updates_saved,registers,shared_bytes,local_bytes,active_blocks_per_sm\n";
        std::vector<unsigned> dims{32,64,128,256,512};
        if(argc>2 && std::string(argv[2])!="all") {int f=std::stoi(argv[2]);require(f>0,"F must be positive");dims={unsigned(f)};}
        for(auto f:dims) {
            auto x=readFeatures(argc>5?argv[5]:nullptr,m.cols,f);Buffer<float> input(x),output(std::size_t(m.rows)*f),baseline(std::size_t(m.rows)*f);
            auto rowCall=[&]() {gr.launch(input.data,baseline.data,f,stream);};rowCall();auto expected=baseline.host(stream);
            auto rowErr=sampledCPU(m,x,expected,f,true);verify(rowErr);auto rowFp=sampledCPU(m,x,expected,f,false);
            float rowMs=measure(rowCall,repeat,stream);int rowAttr[6];check(rowKernelAttributes(rowAttr));
            bool bounded=m.rows%8 || f<64 || f%32 || (f>512 && f%512) || std::uint64_t(m.cols)*f>UINT32_MAX || std::uint64_t(m.rows)*f>UINT32_MAX;
            int a=bounded?3:0;
            std::cout<<f<<",Row,"<<rowConvert<<','<<rowUpload<<','<<rowMs<<",0,"<<rowMs<<",0,0,"<<rowErr.rmse<<','<<rowErr.maxAbsolute
                <<','<<rowFp.rmse<<','<<rowFp.maxAbsolute<<",0,0,0,"<<rowAttr[a]<<','<<rowAttr[a+1]<<','<<rowAttr[a+2]<<",0\n"<<std::flush;
            auto clear=[&]() {check(cudaMemsetAsync(output.data,0,output.size*4,stream));};float clearMs=measure(clear,repeat,stream);
            for(int v=0;v<4;++v) {
                auto kernel=[&]() {check(launchColumnSpMM(gc.view,input.data,output.data,f,ColumnVersion(v),stream,false));};
                auto complete=[&]() {check(launchColumnSpMM(gc.view,input.data,output.data,f,ColumnVersion(v),stream));};
                complete();auto y=output.host(stream);auto rowError=error(y,expected);verify(rowError);
                auto cpu=sampledCPU(m,x,y,f,true);verify(cpu);auto fp=sampledCPU(m,x,y,f,false);double variation=0;
                for(int r=0;r<3;++r) {complete();variation=std::max(variation,error(output.host(stream),y).maxAbsolute);}
                float totalMs=measure(complete,repeat,stream),kernelMs=kernelTime(clear,kernel,repeat,stream);
                verify(error(output.host(stream),expected));ColumnKernelInfo info;check(columnKernelAttributes(ColumnVersion(v),&info));
                auto atomics=(v==3?st.cached:st.uncached)*f, saved=(v==3?st.saved:0)*f;
                std::cout<<f<<",C"<<v<<','<<colConvert<<','<<colUpload<<','<<kernelMs<<','<<clearMs<<','<<totalMs
                    <<','<<rowError.rmse<<','<<rowError.maxAbsolute<<','<<cpu.rmse<<','<<cpu.maxAbsolute<<','<<fp.rmse<<','<<fp.maxAbsolute
                    <<','<<variation<<','<<atomics<<','<<saved<<','<<info.registers<<','<<info.sharedBytes<<','<<info.localBytes<<','<<info.activeBlocksPerSM<<'\n'<<std::flush;
            }
        }
        check(cudaStreamDestroy(stream));
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
