// Correctness + ablation benchmark for the Fast Column kernels.
// Usage:
//   column_fast_test [matrix.csr] [F=128|256] [device=0] [X.bin] [--bench]
// Without --bench: correctness check vs the TF32-rounded CPU reference for
// every FastVariant (V1/V2A/V2B/V2C/V3-2/V3-4/V4-ftile) on a random-ish
// matrix (or the provided CSR). With --bench: full graph+event timing,
// 3 groups, median, per-variant CSV row (task-book fields).
#include "matrix_io.cuh"
#include "column_schedule.cuh"
#include "column_fast.cuh"
#include "column_test_utils.cuh"
#include <algorithm>
#include <iomanip>

using namespace test;
namespace {
using U = std::uint32_t;

struct VariantSpec { const char* name; FastVariant v; U ftile; U stages; };

cudaError_t launchNamed(const DeviceColumnFast& m, const float* x, float* y,
    U features, const VariantSpec& spec, cudaStream_t stream, bool clear) {
    return launchColumnFast(m, x, y, features, spec.v, spec.ftile, spec.stages, stream, clear);
}

struct Result { double med = 0, mn = 0, mx = 0; Error err; ColumnFastInfo info; };
Result measureVariant(const DeviceColumnFast& m, const float* x, float* y, U features,
    const VariantSpec& spec, const std::vector<double>& reference, cudaStream_t stream, bool verifyResult) {
    Result r;
    check(columnFastAttributes(spec.v, spec.ftile, spec.stages, &r.info));
    auto launch = [&]() { check(launchNamed(m, x, y, features, spec, stream, true)); };
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
    for (int l = 0; l < 9; ++l) { // 3 groups x 3 launches
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
    if (verifyResult) {
        auto out = Buffer<float>(std::size_t(m.rows) * features);
        check(cudaMemcpy(out.data, y, std::size_t(m.rows) * features * sizeof(float), cudaMemcpyDeviceToDevice));
        auto host = out.host(stream);
        r.err = error(host, reference);
    }
    return r;
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2 || std::string(argv[1]) == "--help") {
            std::cout << "Usage: column_fast_test matrix.csr [F=128|256] [device=0] [X.bin] [--bench]\n";
            return 0;
        }
        bool bench = false;
        for (int i = 1; i < argc; ++i) if (std::string(argv[i]) == "--bench") bench = true;
        check(cudaSetDevice(argc > 3 ? std::stoi(argv[3]) : 0));
        cudaStream_t stream; check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        auto m = readMatrix(argv[1]);
        const U features = U(argc > 2 ? std::stoi(argv[2]) : 128);
        auto base = buildColumnBase(m.rows, m.cols, m.csr);
        auto x = readFeatures(argc > 4 ? argv[4] : nullptr, m.cols, features);
        Buffer<float> input(x), y(std::size_t(m.rows) * features);
        Buffer<std::uint32_t> win(base.colWindowOffset), rows(base.tileRows), off(base.tileValueOffset);
        Buffer<std::uint64_t> bits(base.tileLocalBit);
        Buffer<float> vals(base.values);
        DeviceColumnFast view{base.rows, base.cols, win.data, rows.data, off.data, bits.data, vals.data};
        auto ref = reference(m, x, features, true);

        const VariantSpec specs[] = {
            {"V1-direct-rows", FAST_V1, 128, 1},
            {"V2A-hoist", FAST_V2A, 128, 1},
            {"V2B-f2-wide", FAST_V2B, 128, 1},
            {"V2C-f4-wide", FAST_V2C, 128, 1},
            {"V3-pipe2", FAST_V3, 128, 2},
            {"V3-pipe4", FAST_V3, 128, 4},
            {"V3W-pipe2-f2", FAST_V3W, 128, 2},
            {"V4-ft32", FAST_V4, 32, 1},
            {"V4-ft64", FAST_V4, 64, 1},
            {"V4-ft256", FAST_V4, 256, 1},
            {"V2C-ft32", FAST_V2C, 32, 1},
            {"V2C-ft64", FAST_V2C, 64, 1},
            {"V2C-ft256", FAST_V2C, 256, 1},
        };
        std::cout << std::setprecision(9);
        if (!bench) {
            std::cout << "# correctness dataset=" << argv[1] << " M=" << m.rows << " K=" << m.cols
                      << " nnz=" << m.csr.src.size() << " F=" << features << '\n';
            std::cout << "variant,relative_rmse,max_absolute_error,registers,shared_bytes,active_blocks_per_sm\n";
            for (auto& s : specs) {
                if (s.v == FAST_V3 && m.rows > 300000) continue; // skip 4-stage on huge graphs for the quick check
                if (s.ftile > features) continue;                 // e.g. V4-ft256 with F=128
                Result r = measureVariant(view, input.data, y.data, features, s, ref, stream, true);
                std::cout << s.name << ',' << r.err.rmse << ',' << r.err.maxAbsolute << ','
                          << r.info.registers << ',' << r.info.sharedBytes << ',' << r.info.activeBlocksPerSM << '\n' << std::flush;
                require(r.err.rmse < 2e-4 && r.err.maxAbsolute < 2e-4 + r.err.maxReference * 5e-4,
                    "Fast kernel TF32 reference mismatch");
            }
            std::cout << "correctness OK\n";
        } else {
            std::cout << "dataset,M,K,nnz,F,variant,kernel_ms,min,max,registers,shared_bytes,local_bytes,active_blocks_per_sm,relative_rmse,max_absolute_error\n";
            for (auto& s : specs) {
                if (s.ftile > features) continue; // e.g. V4-ft256 with F=128
                Result r = measureVariant(view, input.data, y.data, features, s, ref, stream, true);
                std::cout << argv[1] << ',' << m.rows << ',' << m.cols << ',' << m.csr.src.size() << ',' << features
                          << ',' << s.name << ',' << r.med << ',' << r.mn << ',' << r.mx << ','
                          << r.info.registers << ',' << r.info.sharedBytes << ',' << r.info.localBytes << ','
                          << r.info.activeBlocksPerSM << ',' << r.err.rmse << ',' << r.err.maxAbsolute << '\n' << std::flush;
            }
        }
        check(cudaStreamDestroy(stream));
    } catch (const std::exception& e) { std::cerr << "Error: " << e.what() << '\n'; return 1; }
    return 0;
}
