#include "matrix_io.cuh"
#include "../tac_bcsc/ptx_compat.cuh"
// Same write-through store instruction as the original TAC helper.
__device__ __forceinline__ void store_fp32_to_global(float* ptr,float value) {
    asm volatile("st.global.wt.f32 [%0], %1;" :: "l"(ptr),"f"(value):"memory");
}
#include "original_kernel.cuh"
#include <iomanip>
using namespace test;
struct Format {
    std::vector<unsigned> windows{0}, offsets{0}, rows;
    std::vector<uint64_t> bits;
    unsigned empty=0;
};
// TAC COO2BCSR layout: 8 fixed destination rows, sorted unique source
// columns grouped into tiles of 8; bit = local destination row * 8 + source slot.
Format convert(const Matrix& m) {
    Format out;
    for(unsigned rw=0;rw<(m.rows+7)/8;++rw) {
        std::vector<uint64_t> edges;
        for(unsigned r=rw*8;r<std::min(m.rows,rw*8+8);++r)
            for(unsigned e=m.csr.ptr[r];e<m.csr.ptr[r+1];++e) {
                require(m.csr.values[e]==1.f,"TAC kernel supports unit weights only");
                edges.push_back((uint64_t(m.csr.src[e])<<3)|(r%8));
            }
        std::sort(edges.begin(),edges.end());
        unsigned previous=UINT32_MAX,slot=0;
        size_t begin=out.bits.size();
        for(auto edge:edges) {
            unsigned source=edge>>3,localRow=edge&7;
            if(source!=previous) {
                if(slot%8==0) {out.bits.push_back(0);out.rows.resize(out.rows.size()+8,UINT32_MAX);}
                out.rows[(out.bits.size()-1)*8+slot%8]=source;
                previous=source;++slot;
            }
            out.bits.back()|=1ULL<<(localRow*8+(slot-1)%8);
        }
        // Unconditional first-tile prefetch; UINT32_MAX also avoids upstream
        // next-tile checks using > numCols instead of >= numCols.
        if(begin==out.bits.size()) {++out.empty;out.bits.push_back(0);out.rows.resize(out.rows.size()+8,UINT32_MAX);}
        for(size_t t=begin;t<out.bits.size();++t) out.offsets.push_back(out.offsets.back()+__builtin_popcountll(out.bits[t]));
        out.windows.push_back(out.bits.size());
    }
    return out;
}
__global__ void referenceCSR(const unsigned* ptr,const unsigned* col,const float* x,float* y,unsigned rows,unsigned f) {
    size_t idx=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(idx>=size_t(rows)*f)return;
    unsigned r=idx/f,j=idx%f;double sum=0;
    for(unsigned e=ptr[r];e<ptr[r+1];++e) {
        float v=x[size_t(col[e])*f+j];
        sum+=double(__uint_as_float(rounded_tf32(v)));
    }
    y[idx]=float(sum);
}
void run(const std::string& name,const Matrix& m,unsigned f,const char* featurePath=nullptr,bool cpu=false) {
    require(f>=64 && f<=512 && f%32==0,"F must be a multiple of 32 in [64,512]");
    require(m.rows && m.cols,"matrix dimensions must be nonzero");
    auto start=Clock::now();auto fmt=convert(m);double conversion=elapsed(start);
    Buffer<unsigned> windows(fmt.windows),offsets(fmt.offsets),rows(fmt.rows),ptr(m.csr.ptr),col(m.csr.src);
    Buffer<uint64_t> bits(fmt.bits);
    auto x=readFeatures(featurePath,m.cols,f);
    Buffer<float> dx(x),dy(size_t((m.rows+7)/8*8)*f),ref(size_t(m.rows)*f);
    cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    auto kernel=[&]{tf32_computeX128TransposePipe2_BCSR_forward_first_layer<<<dim3(fmt.windows.size()-1),dim3(32,f/32),0,stream>>>(windows.data,offsets.data,rows.data,bits.data,dx.data,dy.data,m.rows,m.cols,f);check(cudaGetLastError());};
    auto complete=[&]{check(cudaMemsetAsync(dy.data,0,dy.size*sizeof(float),stream));kernel();};
    check(cudaMemsetAsync(dy.data,0xff,dy.size*sizeof(float),stream));
    kernel();auto output=dy.host(stream);output.resize(size_t(m.rows)*f);
    referenceCSR<<<(ref.size+255)/256,256,0,stream>>>(ptr.data,col.data,dx.data,ref.data,m.rows,f);
    check(cudaGetLastError());auto expected=ref.host(stream);auto err=error(output,expected);verify(err);
    if(cpu)verify(error(output,reference(m,x,f,true)));
    // Row kernel must overwrite all output, without a clear, including empty rows.
    kernel();auto repeated=dy.host(stream);repeated.resize(expected.size());verify(error(repeated,expected));
    float completeMs=measure(complete,10,stream);
    check(cudaMemsetAsync(dy.data,0,dy.size*sizeof(float),stream));
    float kernelMs=measure(kernel,10,stream);
    std::cout<<name<<','<<m.rows<<','<<m.cols<<','<<m.csr.src.size()<<','<<f<<','<<fmt.bits.size()<<','<<fmt.empty<<','<<conversion<<','<<kernelMs<<','<<completeMs<<','<<err.rmse<<','<<err.maxAbsolute<<'\n'<<std::flush;
    check(cudaStreamDestroy(stream));
}
int main(int argc,char** argv) {
 try {
    check(cudaSetDevice(0));std::cout<<std::setprecision(9);
    std::cout<<"matrix,rows,cols,nnz,F,tiles,empty_windows,convert_ms,kernel_ms,clear_kernel_ms,relative_l2,max_abs\n";
    if(argc==2 && std::string(argv[1])=="--test") {
        for(unsigned f:{64,96,128,256,512}) {
            run("dense_tail",makeMatrix(19,21,[](unsigned,unsigned){return 1.f;}),f,nullptr,true);
            run("empty",makeMatrix(17,25,[](unsigned,unsigned){return 0.f;}),f,nullptr,true);
            run("empty_windows_tail",makeMatrix(41,49,[](unsigned r,unsigned c){return (r<8||r==40) && (r+c)%3==0?1.f:0.f;}),f,nullptr,true);
            run("single",makeMatrix(1,1,[](unsigned,unsigned){return 1.f;}),f,nullptr,true);
            run("random",makeMatrix(137,73,[](unsigned r,unsigned c){return ((r*177+c*23+r*c)%31)<5?1.f:0.f;}),f,nullptr,true);
        }
        bool rejected=false;try{convert(makeMatrix(1,1,[](unsigned,unsigned){return 2.f;}));}catch(const std::exception&){rejected=true;}
        require(rejected,"weighted input must be rejected");
    } else {
        require(argc>=3,"usage: tac_bcsr_repro --test | matrix.csr F [X.bin]");
        run(argv[1],readMatrix(argv[1]),std::stoul(argv[2]),argc>3?argv[3]:nullptr);
    }
 } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
