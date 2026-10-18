#include "matrix_io.cuh"
#include "ptx_compat.cuh"
#include "original_kernel.cuh"
#include <iomanip>
using namespace test;
struct Format {
    std::vector<unsigned> windows{0}, offsets{0}, rows;
    std::vector<uint64_t> bits;
    unsigned empty=0;
};
// Same layout as TAC COO2BCSC: windows of 8 source columns, sorted unique
// destination rows, 8 rows per tile, row-major bitmap. Unit edge weights only.
Format convert(const Matrix& m) {
    Format out;
    std::vector<uint64_t> edges;
    edges.reserve(m.csr.src.size());
    for(unsigned r=0;r<m.rows;++r) for(unsigned e=m.csr.ptr[r];e<m.csr.ptr[r+1];++e) {
        require(m.csr.values[e]==1.f,"TAC kernel supports unit weights only");
        edges.push_back((uint64_t(m.csr.src[e]/8)<<35)|(uint64_t(r)<<3)|(m.csr.src[e]%8));
    }
    std::sort(edges.begin(),edges.end());
    size_t pos=0;
    for(unsigned cw=0;cw<(m.cols+7)/8;++cw) {
        unsigned previous=UINT32_MAX,slot=0;
        size_t begin=out.bits.size();
        while(pos<edges.size() && (edges[pos]>>35)==cw) {
            unsigned row=(edges[pos]>>3)&UINT32_MAX, col=edges[pos]&7;
            if(row!=previous) {
                if(slot%8==0) {out.bits.push_back(0);out.rows.resize(out.rows.size()+8,UINT32_MAX);}
                out.rows[(out.bits.size()-1)*8+slot%8]=row;
                previous=row;++slot;
            }
            out.bits.back()|=1ULL<<(((slot-1)%8)*8+col);
            ++pos;
        }
        // Original kernel unconditionally prefetches its first tile.
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
    Buffer<float> dx(x),dy(size_t(m.rows)*f),ref(size_t(m.rows)*f);
    cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    auto kernel=[&]{tf32_computeX128TransposePipe2_BCSC_forward_first_layer<<<dim3(fmt.windows.size()-1),dim3(32,f/32),0,stream>>>(windows.data,offsets.data,rows.data,bits.data,dx.data,dy.data,m.rows,m.cols,f);check(cudaGetLastError());};
    auto complete=[&]{check(cudaMemsetAsync(dy.data,0,dy.size*sizeof(float),stream));kernel();};
    complete();auto output=dy.host(stream);
    referenceCSR<<<(ref.size+255)/256,256,0,stream>>>(ptr.data,col.data,dx.data,ref.data,m.rows,f);
    check(cudaGetLastError());auto expected=ref.host(stream);auto err=error(output,expected);verify(err);
    if(cpu)verify(error(output,reference(m,x,f,true)));
    // Explicitly check output reset across repeated independent calls.
    complete();verify(error(dy.host(stream),expected));
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
            run("empty_windows_tail",makeMatrix(41,49,[](unsigned r,unsigned c){return (c<8||c==48) && (r+c)%3==0?1.f:0.f;}),f,nullptr,true);
            run("single",makeMatrix(1,1,[](unsigned,unsigned){return 1.f;}),f,nullptr,true);
            run("random",makeMatrix(137,73,[](unsigned r,unsigned c){return ((r*177+c*23+r*c)%31)<5?1.f:0.f;}),f,nullptr,true);
        }
        bool rejected=false;try{convert(makeMatrix(1,1,[](unsigned,unsigned){return 2.f;}));}catch(const std::exception&){rejected=true;}
        require(rejected,"weighted input must be rejected");
    } else {
        require(argc>=3,"usage: tac_bcsc_repro --test | matrix.csr F [X.bin]");
        run(argv[1],readMatrix(argv[1]),std::stoul(argv[2]),argc>3?argv[3]:nullptr);
    }
 } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
