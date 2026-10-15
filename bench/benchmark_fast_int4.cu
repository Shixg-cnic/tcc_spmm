// Batch 4: best FP32 Column kernels converted to fused INT4 X, plus the
// Acc Basic INT4 Fused baseline. All methods run on the same CSR/X/F/GPU/
// stream with the same graph+event timing (kernel-only median, >=3 groups).
//
// Methods:
//   acc_fp32       : original Acc-SpMM row kernel on the dequantised X-hat
//   acc_int4_fused : Acc kernel with fused INT4 X loads (batch 4 new)
//   fast_v2c_fp32  : Fast V2C on X-hat (same structure, FP32 X)
//   fast_v2c_int4  : Fast V2C with fused INT4 X (products best -> INT4)
//   fast_v3w_int4  : Fast V3W(p2) with fused INT4 X (arxiv best -> INT4)
//   c0_int4        : fixed C0 kernel with fused INT4 X (batch-0 infra)
//
// Correctness vs a sampled CPU TF32 reference on X-hat (asserted) plus the
// quantization delta vs the reference on the ORIGINAL X (reported, not
// asserted). The FP32-side methods use the GPU-dequantised X-hat so the INT4
// vs FP32 comparison isolates the fused-load delta from the dequant pipeline.
#include "matrix_io.cuh"
#include "column_schedule.cuh"
#include "column_fast.cuh"
#include "column_test_utils.cuh"
#include "int4_format.cuh"
#include "acc_reference.cuh"
#include <iomanip>

using namespace test;
using U = std::uint32_t;

// launcher from bench/acc_int4_fused.cu (mechanical copy of the Acc kernel
// with fused INT4 X loads; the third_party original is unmodified).
// Defined at global scope in that TU (single-TU Acc unit, see CMake).
cudaError_t launchAccRowInt4SpMM(const std::uint32_t* window, const std::uint32_t* offset,
    const std::uint32_t* mapping, const std::uint64_t* bits, const float* values,
    const std::uint8_t* xq, const float* xscale, float* y, std::uint32_t paddedRows,
    std::uint32_t cols, std::uint32_t features, cudaStream_t stream);

namespace {

// Sampled TF32 reference over 64 spread rows (same convention as
// benchmark_fp32 / benchmark_int4).
Error sampledCPU(const Matrix& m, const std::vector<float>& x, const std::vector<float>& y,
    unsigned f, bool rounded) {
    std::vector<double> ref; std::vector<float> sampled;
    const unsigned count = std::min(64u, m.rows);
    for (unsigned i = 0; i < count; ++i) {
        auto row = std::uint64_t(i) * (m.rows - 1) / std::max(1u, count - 1);
        for (unsigned j = 0; j < f; ++j) {
            double sum = 0;
            for (auto e = m.csr.ptr[row]; e < m.csr.ptr[row + 1]; ++e) {
                float a = m.csr.values[e], b = x[std::size_t(m.csr.src[e]) * f + j];
                sum += double(rounded ? tf32(a) : a) * (rounded ? tf32(b) : b);
            }
            ref.push_back(sum); sampled.push_back(y[row * f + j]);
        }
    }
    return error(sampled, ref);
}

struct Result { double med = 0, mn = 0, mx = 0; Error errXhat; Error errX; };

Result measureKernel(const std::function<void()>& launch, cudaStream_t stream) {
    Result r;
    for (int i = 0; i < 10; ++i) launch();
    check(cudaStreamSynchronize(stream));
    cudaGraph_t g; cudaGraphExec_t exec;
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    const int repeat = 5;
    for (int i = 0; i < repeat; ++i) launch();
    check(cudaStreamEndCapture(stream, &g));
    check(cudaGraphInstantiate(&exec, g, nullptr, nullptr, 0));
    for (int w = 0; w < 5; ++w) { check(cudaGraphLaunch(exec, stream)); check(cudaStreamSynchronize(stream)); }
    cudaEvent_t begin, end;
    check(cudaEventCreate(&begin)); check(cudaEventCreate(&end));
    std::vector<float> samples;
    for (int l = 0; l < 9; ++l) {
        check(cudaEventRecord(begin, stream));
        check(cudaGraphLaunch(exec, stream));
        check(cudaEventRecord(end, stream));
        check(cudaEventSynchronize(end));
        float ms; check(cudaEventElapsedTime(&ms, begin, end));
        samples.push_back(ms / repeat);
    }
    std::sort(samples.begin(), samples.end());
    r.med = samples[samples.size() / 2]; r.mn = samples.front(); r.mx = samples.back();
    cudaEventDestroy(begin); cudaEventDestroy(end);
    cudaGraphExecDestroy(exec); cudaGraphDestroy(g);
    return r;
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2 || std::string(argv[1]) == "--help") {
            std::cout << "Usage: benchmark_fast_int4 matrix.csr [F=128|256] [device=0] [X.bin] [--no-verify]\n";
            return 0;
        }
        bool verify = true;
        for (int i = 1; i < argc; ++i) if (std::string(argv[i]) == "--no-verify") verify = false;
        check(cudaSetDevice(argc > 3 ? std::stoi(argv[3]) : 0));
        cudaStream_t stream; check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        auto m = readMatrix(argv[1]);
        const U features = U(argc > 2 ? std::stoi(argv[2]) : 128);
        auto col = convertColumn(m.rows, m.cols, m.csr);
        auto base = buildColumnBase(m.rows, m.cols, m.csr);
        auto x = readFeatures(argc > 4 ? argv[4] : nullptr, m.cols, features);

        // ---- formats (Column fast view, fixed-C0 view, Acc row) ----
        Buffer<float> input(x), y(std::size_t(m.rows) * features);
        Buffer<std::uint32_t> win(base.colWindowOffset), rows(base.tileRows), off(base.tileValueOffset);
        Buffer<std::uint64_t> bits(base.tileLocalBit);
        Buffer<float> vals(base.values);
        DeviceColumnFast view{base.rows, base.cols, win.data, rows.data, off.data, bits.data, vals.data};
        GPUColumn gc(col);

        RowBitTCF accF = convertAccRow(m.rows, m.cols, m.csr);
        Buffer<std::uint64_t> accBits(accF.tcLocalBit);
        Buffer<std::uint32_t> accWindow(accF.rowWindowOffset), accOffset(accF.tcOffset),
            accMapping(accF.sparseA2B);
        Buffer<float> accValues(accF.values);
        const U paddedRows = U((m.rows + 7) / 8 * 8);

        // ---- INT4 X: quantize once; dequantise to X-hat for the FP32-side runs ----
        const U pairs = (features + 1) / 2;
        Buffer<std::uint8_t> xq(std::size_t(m.cols) * pairs);
        Buffer<float> xscale(m.cols);
        check(quantizeXInt4(input.data, xq.data, xscale.data, m.cols, features, stream));
        Buffer<float> xhat(std::size_t(m.cols) * features);
        check(dequantizeXInt4(xq.data, xscale.data, xhat.data, m.cols, features, stream));
        check(cudaStreamSynchronize(stream));
        DeviceInt4X int4x{xq.data, xscale.data};

        // ---- methods ----
        struct Spec { const char* name; std::function<void()> launch; };
        const std::size_t yBytes = std::size_t(m.rows) * features * sizeof(float);
        std::vector<Spec> specs;
        specs.push_back({"acc_fp32", [&]() {
            check(cudaMemsetAsync(y.data, 0, yBytes, stream));
            check(launchAccRowSpMM(accWindow.data, accOffset.data, accMapping.data, accBits.data,
                accValues.data, xhat.data, y.data, paddedRows, m.cols, features, stream));
        }});
        specs.push_back({"acc_int4_fused", [&]() {
            check(cudaMemsetAsync(y.data, 0, yBytes, stream));
            check(launchAccRowInt4SpMM(accWindow.data, accOffset.data, accMapping.data, accBits.data,
                accValues.data, xq.data, xscale.data, y.data, paddedRows, m.cols, features, stream));
        }});
        specs.push_back({"fast_v2c_fp32", [&]() {
            check(launchColumnFast(view, xhat.data, y.data, features, FAST_V2C, 128, 1, stream, true));
        }});
        specs.push_back({"fast_v2c_int4", [&]() {
            check(launchColumnFastInt4(view, int4x, y.data, features, FAST_V2C, 128, 1, stream, true));
        }});
        specs.push_back({"fast_v3w_int4", [&]() {
            check(launchColumnFastInt4(view, int4x, y.data, features, FAST_V3W, 128, 2, stream, true));
        }});
        specs.push_back({"c0_int4", [&]() {
            check(cudaMemsetAsync(y.data, 0, yBytes, stream));
            check(launchColumnInt4SpMM(gc.view, int4x, y.data, features, ColumnVersion::C0, stream, false));
        }});

        std::cout << std::setprecision(9);
        std::cout << "dataset,M,K,nnz,F,method,kernel_ms,min,max,rmse_xhat,maxAbs_xhat,rmse_x,maxAbs_x\n" << std::flush;
        for (auto& s : specs) {
            Result r = measureKernel(s.launch, stream);
            Error eXhat{0, 0, 0}, eX{0, 0, 0};
            if (verify) {
                auto yHost = y.host(stream);
                std::vector<float> valid(yHost.begin(), yHost.begin() + std::size_t(m.rows) * features);
                eXhat = sampledCPU(m, xhat.host(stream), valid, features, true);
                eX = sampledCPU(m, x, valid, features, false);
                require(eXhat.rmse < 2e-4 && eXhat.maxAbsolute < 2e-4 + eXhat.maxReference * 5e-4,
                    (std::string("TF32-on-Xhat reference mismatch for ") + s.name).c_str());
            }
            std::cout << argv[1] << ',' << m.rows << ',' << m.cols << ',' << m.csr.src.size() << ',' << features
                      << ',' << s.name << ',' << r.med << ',' << r.mn << ',' << r.mx << ','
                      << eXhat.rmse << ',' << eXhat.maxAbsolute << ','
                      << eX.rmse << ',' << eX.maxAbsolute << '\n' << std::flush;
        }
        check(cudaStreamDestroy(stream));
    } catch (const std::exception& e) { std::cerr << "Error: " << e.what() << '\n'; return 1; }
    return 0;
}
