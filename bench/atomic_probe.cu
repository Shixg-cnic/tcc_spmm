// Diagnostic probe: identical C0 column flow but the MMA output is discarded
// (accumulated into a tiny per-CTA sink register) instead of issued as global
// atomicAdd. Running this next to the real C0 isolates the cost of the
// atomic output path from the MMA/load/pipeline cost. NOT a production kernel.
#include "matrix_io.cuh"
#include "mma_tf32.cuh"
#include "pipeline.cuh"
#include <iomanip>

using namespace test;
namespace {
using U = std::uint32_t;

__device__ void loadX(const float* x, U w, U cols, U f, U features, float b[2][4]) {
    U lane=threadIdx.x%32, group=lane/4, k=lane%4;
    #pragma unroll
    for (int part=0; part<2; ++part) {
        #pragma unroll
        for (int i=0; i<4; ++i) {
            U col=w*8+k+(i>=2 ? 4 : 0);
            U feature=f+group+(i%2)*8+part*16;
            b[part][i]=col<cols && feature<features ? x[std::size_t(col)*features+feature] : 0.f;
        }
    }
}
__device__ void stageTile(DeviceColumnSWTCF m, U tile, float* a, U* slots) {
    U tid=threadIdx.x;
    if (tid<64) {
        auto bitmap=m.tileLocalBit[tile];
        a[tid]=(bitmap & (1ULL<<tid)) ?
            m.values[m.tileValueOffset[tile]+__popcll(bitmap & ((1ULL<<tid)-1))] : 0.f;
    }
    if (tid<8) slots[tid]=m.tileRowSlot[std::size_t(tile)*8+tid];
}

__global__ void columnProbeNoAtomic(DeviceColumnSWTCF m, const float* x, float* sink, U features) {
    __shared__ float a[64];
    __shared__ U slots[8];
    const U windows=(std::uint64_t(m.cols)+7)/8;
    U firstWindow=blockIdx.x;   // C0: one CW per CTA
    U sw=firstWindow/m.superWindowSize;
    U rowBase=m.superWindowRowOffset[sw];
    U fBase=blockIdx.y*COLUMN_FEATURE_TILE+(threadIdx.x/32)*32;
    float b[2][4];
    float sinkAcc=0.f;
    for(U w=firstWindow;w<firstWindow+1;++w) {
        loadX(x,w,m.cols,fBase,features,b);
        U first=m.colWindowOffset[w], end=m.colWindowOffset[w+1];
        for(U tile=first;tile<end;++tile) {
            stageTile(m,tile,a,slots);
            __syncthreads();
            U lane=threadIdx.x%32, group=lane/4, k=lane%4;
            float sparse[2]={a[group*8+k],a[group*8+k+4]};
            #pragma unroll
            for(int part=0;part<2;++part) {
                float c[4]={0.f,0.f,0.f,0.f};
                tf32_m16n8k8(b[part],sparse,c);
                #pragma unroll
                for(int i=0;i<4;++i) {
                    U slot=slots[k*2+i%2];
                    U feature=fBase+group+(i>=2 ? 8 : 0)+part*16;
                    if(slot!=UINT32_MAX && feature<features) sinkAcc+=c[i];  // no atomic
                }
            }
            __syncthreads();
        }
    }
    if (sinkAcc==12345.678f) sink[0]=sinkAcc;  // defeat DCE, never true
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) { std::cerr << "Usage: atomic_probe matrix.csr [F=128] [groups=3] [device=0] [X.bin]\n"; return 1; }
        int groups = argc > 3 ? std::stoi(argv[3]) : 3;
        check(cudaSetDevice(argc > 4 ? std::stoi(argv[4]) : 0));
        cudaStream_t stream; check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        auto m = readMatrix(argv[1]);
        auto col = convertColumn(m.rows, m.cols, m.csr);
        GPUColumn gc(col);
        unsigned f = argc > 2 ? unsigned(std::stoi(argv[2])) : 128;
        auto x = readFeatures(argc > 5 ? argv[5] : nullptr, m.cols, f);
        Buffer<float> input(x), y(std::size_t(m.rows) * f), sink(32);
        auto launch = [&]() { columnProbeNoAtomic<<<dim3((std::uint64_t(m.cols)+7)/8,
            (std::uint64_t(f)+COLUMN_FEATURE_TILE-1)/COLUMN_FEATURE_TILE), dim3(COLUMN_FEATURE_TILE), 0, stream>>>(
            gc.view, input.data, sink.data, f); check(cudaGetLastError()); };
        launch(); check(cudaStreamSynchronize(stream));
        // timing: same graph methodology as benchmark_fp32 (complete = memset+kernel)
        auto pair = [&]() {
            std::vector<float> completes, kernels;
            std::vector<cudaEvent_t> start(3), stop(3);
            for (int i=0;i<3;++i) { check(cudaEventCreate(&start[i])); check(cudaEventCreate(&stop[i])); }
            for (int g=0; g<groups; ++g) {
                cudaGraph_t graph; cudaGraphExec_t exec;
                check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
                for (int i=0;i<3;++i) {
                    check(cudaMemsetAsync(y.data,0,std::size_t(m.rows)*f*sizeof(float),stream));
                    check(cudaEventRecordWithFlags(start[i],stream,cudaEventRecordExternal));
                    launch();
                    check(cudaEventRecordWithFlags(stop[i],stream,cudaEventRecordExternal));
                }
                check(cudaStreamEndCapture(stream,&graph));
                check(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
                for (int w=0;w<5;++w) { check(cudaGraphLaunch(exec,stream)); check(cudaStreamSynchronize(stream)); }
                cudaEvent_t begin,end;
                check(cudaEventCreate(&begin)); check(cudaEventCreate(&end));
                for (int l=0;l<3;++l) {
                    check(cudaEventRecord(begin,stream));
                    check(cudaGraphLaunch(exec,stream));
                    check(cudaEventRecord(end,stream));
                    check(cudaEventSynchronize(end));
                    float total; check(cudaEventElapsedTime(&total,begin,end));
                    float kernel=0;
                    for (int i=0;i<3;++i) { float t; check(cudaEventElapsedTime(&t,start[i],stop[i])); kernel+=t; }
                    completes.push_back(total/3); kernels.push_back(kernel/3);
                }
                cudaEventDestroy(begin); cudaEventDestroy(end);
                cudaGraphExecDestroy(exec); cudaGraphDestroy(graph);
            }
            for (int i=0;i<3;++i) { cudaEventDestroy(start[i]); cudaEventDestroy(stop[i]); }
            auto med=[](std::vector<float>& v){ std::sort(v.begin(),v.end()); return v[v.size()/2]; };
            return std::make_pair(med(kernels), med(completes));
        }();
        std::cout << std::setprecision(9) << "probe_no_atomic,dataset=" << argv[1] << " M=" << m.rows
                  << " K=" << m.cols << " nnz=" << m.csr.src.size() << " F=" << f
                  << " kernel_ms=" << pair.first << " complete_ms=" << pair.second
                  << " (real C0 kernel_ms for comparison from benchmark_fp32)\n";
        check(cudaStreamDestroy(stream));
    } catch (const std::exception& e) { std::cerr << "Error: " << e.what() << '\n'; return 1; }
    return 0;
}
