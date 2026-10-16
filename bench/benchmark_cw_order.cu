// CW execution-order experiment benchmark (experiment/cw-order-only).
// Compares only Acc_Basic vs the V2C family on the same CSR/X/F/GPU/stream:
//   Acc_Basic        : original Acc-SpMM row kernel (unmodified)
//   V2C_Original     : current unordered V2C
//   V2C_IdentityOrder: V2C through the identity task-order mapping
//   V2C_RandomOrder  : V2C through the fixed-seed random mapping
//   V2C_HotAnchor    : V2C through the hot-output-row anchor mapping
//   V2C_BandAnchor   : V2C through the dominant-band anchor mapping
//
// Timing (same口径 as benchmark_fp32): CUDA Graph + Events, >=3 capture
// groups, 5 repeats per group, 5 warmups; per-iteration median reported.
// Kernel-only: memset(Y) then events AROUND the kernel only. Complete:
// memset(Y) + kernel. Acc uses the same init rules; additionally reports its
// no-clear kernel interval.
//
// Correctness: full Y of every method compared against the full Acc output
// (same tolerance as the project: rmse<2e-4, maxAbs<2e-4+maxRef*5e-4).
#include "matrix_io.cuh"
#include "acc_reference.cuh"
#include "column_schedule.cuh"
#include "column_fast.cuh"
#include "cw_order.cuh"
#include <iomanip>

using namespace test;
namespace {
using U = std::uint32_t;

// Acc writes paddedRows*F; every method shares a padded Y buffer and all
// correctness comparisons truncate to the real m.rows*F values.
inline std::vector<float> head(std::vector<float> v, std::size_t n) { v.resize(n); return v; }

struct Stats { float med = 0, mn = 0, mx = 0; };
Stats medMinMax(std::vector<float>& v) {
    std::sort(v.begin(), v.end());
    return {v[v.size() / 2], v.front(), v.back()};
}

// Graph + events; per iteration: memset(Y) then external events around the
// kernel (launch must NOT memset). Returns complete (memset+kernel) and
// isolated-kernel intervals from the same launches.
struct PairStats { Stats complete, kernel; };
PairStats measureColumn(const std::function<void()>& launch, void* y, std::size_t bytes,
    int repeat, int groups, int launches, cudaStream_t stream) {
    std::vector<cudaEvent_t> start(repeat), stop(repeat);
    for (int i = 0; i < repeat; ++i) { check(cudaEventCreate(&start[i])); check(cudaEventCreate(&stop[i])); }
    std::vector<float> completes, kernels;
    for (int g = 0; g < groups; ++g) {
        cudaGraph_t graph; cudaGraphExec_t exec;
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        for (int i = 0; i < repeat; ++i) {
            check(cudaMemsetAsync(y, 0, bytes, stream));
            check(cudaEventRecordWithFlags(start[i], stream, cudaEventRecordExternal));
            launch();
            check(cudaEventRecordWithFlags(stop[i], stream, cudaEventRecordExternal));
        }
        check(cudaStreamEndCapture(stream, &graph));
        check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        for (int w = 0; w < 5; ++w) { check(cudaGraphLaunch(exec, stream)); check(cudaStreamSynchronize(stream)); }
        cudaEvent_t begin, end;
        check(cudaEventCreate(&begin)); check(cudaEventCreate(&end));
        for (int l = 0; l < launches; ++l) {
            check(cudaEventRecord(begin, stream));
            check(cudaGraphLaunch(exec, stream));
            check(cudaEventRecord(end, stream));
            check(cudaEventSynchronize(end));
            float t; check(cudaEventElapsedTime(&t, begin, end)); completes.push_back(t / repeat);
        }
        cudaEventDestroy(begin); cudaEventDestroy(end);
        for (int i = 0; i < repeat; ++i) {
            float k; check(cudaEventElapsedTime(&k, start[i], stop[i])); kernels.push_back(k);
        }
        cudaGraphExecDestroy(exec); cudaGraphDestroy(graph);
    }
    for (int i = 0; i < repeat; ++i) { cudaEventDestroy(start[i]); cudaEventDestroy(stop[i]); }
    return {medMinMax(completes), medMinMax(kernels)};
}

Stats measureGraph(const std::function<void()>& operation, int repeat,
    int groups, int launches, cudaStream_t stream) {
    std::vector<float> samples;
    for (int g = 0; g < groups; ++g) {
        cudaGraph_t graph; cudaGraphExec_t exec;
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        for (int i = 0; i < repeat; ++i) operation();
        check(cudaStreamEndCapture(stream, &graph));
        check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        for (int w = 0; w < 5; ++w) { check(cudaGraphLaunch(exec, stream)); check(cudaStreamSynchronize(stream)); }
        cudaEvent_t begin, end;
        check(cudaEventCreate(&begin)); check(cudaEventCreate(&end));
        for (int l = 0; l < launches; ++l) {
            check(cudaEventRecord(begin, stream));
            check(cudaGraphLaunch(exec, stream));
            check(cudaEventRecord(end, stream));
            check(cudaEventSynchronize(end));
            float ms; check(cudaEventElapsedTime(&ms, begin, end)); samples.push_back(ms / repeat);
        }
        cudaEventDestroy(begin); cudaEventDestroy(end);
        cudaGraphExecDestroy(exec); cudaGraphDestroy(graph);
    }
    return medMinMax(samples);
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2 || std::string(argv[1]) == "--help") {
            std::cout << "Usage: benchmark_cw_order matrix.csr F device X.bin bandRows\n";
            return 0;
        }
        const int groups = 3, repeat = 5, launches = 3;
        check(cudaSetDevice(argc > 3 ? std::stoi(argv[3]) : 0));
        const U bandRows = U(argc > 5 ? std::stoi(argv[5]) : 16384);
        cudaStream_t stream; check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        auto m = readMatrix(argv[1]);
        const U features = U(std::stoi(argv[2]));
        require(features == 128 || features == 256, "F must be 128 or 256");
        const U paddedRows = U((std::uint64_t(m.rows) + 7) / 8 * 8);
        const std::uint32_t groupSizes[3] = {32, 64, 128};

        auto base = buildColumnBase(m.rows, m.cols, m.csr);
        auto x = readFeatures(argc > 4 ? argv[4] : nullptr, m.cols, features);
        auto t0 = Clock::now();
        RowBitTCF accF = convertAccRow(m.rows, m.cols, m.csr);
        double accConvertMs = elapsed(t0);

        Buffer<float> input(x), y(std::size_t(paddedRows) * features);
        Buffer<std::uint32_t> win(base.colWindowOffset), rows(base.tileRows), off(base.tileValueOffset);
        Buffer<std::uint64_t> bits(base.tileLocalBit);
        Buffer<float> vals(base.values);
        DeviceColumnFast view{base.rows, base.cols, win.data, rows.data, off.data, bits.data, vals.data};
        const std::size_t real = std::size_t(m.rows) * features;

        Buffer<std::uint64_t> accBits(accF.tcLocalBit);
        Buffer<std::uint32_t> accWindow(accF.rowWindowOffset), accOffset(accF.tcOffset),
            accMapping(accF.sparseA2B);
        Buffer<float> accValues(accF.values);

        // ---- offline orders (all four; BandAnchor at the requested bandRows) ----
        const U W = U(base.colWindowOffset.size() - 1);
        struct OrderEntry { const char* name; CWOrderStrategy strategy; CWOrderResult res; CWOrderStructureStats st; Buffer<U>* d; };
        std::vector<OrderEntry> entries;
        auto buildOne = [&](const char* name, CWOrderStrategy strat) {
            OrderEntry e{name, strat, buildCWOrder(base, CWOrderConfig{strat, bandRows, 20261016}), {}, nullptr};
            validateCWOrder(e.res.order, W);
            e.st = analyzeCWOrder(base, e.res.order, bandRows, groupSizes);
            e.d = new Buffer<U>(e.res.order);
            entries.push_back(std::move(e));
        };
        buildOne("identity", CWOrderStrategy::Identity);
        buildOne("random", CWOrderStrategy::SeededRandom);
        buildOne("hot", CWOrderStrategy::HotAnchor);
        buildOne("band", CWOrderStrategy::BandAnchor);
        // determinism: rebuild once and compare
        for (auto& e : entries) {
            auto r2 = buildCWOrder(base, CWOrderConfig{e.strategy, bandRows, 20261016});
            require(r2.order == e.res.order, ("taskOrder not deterministic for " + std::string(e.name)).c_str());
        }
        const double preprocessMs = entries[0].res.buildMs + entries[1].res.buildMs +
            entries[2].res.buildMs + entries[3].res.buildMs;

        // upload order arrays (one-time, outside the timed loops)
        t0 = Clock::now();
        for (auto& e : entries) check(cudaMemcpy(e.d->data, e.res.order.data(), W * 4, cudaMemcpyHostToDevice));
        check(cudaStreamSynchronize(stream));
        double uploadOrderMs = elapsed(t0);

        // ---- reference: full Acc output ----
        {
            check(cudaMemsetAsync(y.data, 0, std::size_t(paddedRows) * features * sizeof(float), stream));
            check(launchAccRowSpMM(accWindow.data, accOffset.data, accMapping.data, accBits.data,
                accValues.data, input.data, y.data, paddedRows, m.cols, features, stream));
            check(cudaStreamSynchronize(stream));
        }
        auto accOut = y.host(stream);

        // ---- methods ----
        const std::size_t yBytes = std::size_t(paddedRows) * features * sizeof(float);
        struct Method { const char* name; U bandRows; std::function<void()> launch; };
        std::vector<Method> methods;
        methods.push_back({"Acc_Basic", 0, [&]() {
            check(launchAccRowSpMM(accWindow.data, accOffset.data, accMapping.data, accBits.data,
                accValues.data, input.data, y.data, paddedRows, m.cols, features, stream));
        }});
        methods.push_back({"V2C_Original", 0, [&]() {
            check(launchColumnFast(view, input.data, y.data, features, FAST_V2C, 128, 1, stream, false));
        }});
        methods.push_back({"V2C_IdentityOrder", 0, [&]() {
            check(launchColumnFastOrdered(view, input.data, y.data, entries[0].d->data, features, stream, false));
        }});
        methods.push_back({"V2C_RandomOrder", 0, [&]() {
            check(launchColumnFastOrdered(view, input.data, y.data, entries[1].d->data, features, stream, false));
        }});
        methods.push_back({"V2C_HotAnchor", 0, [&]() {
            check(launchColumnFastOrdered(view, input.data, y.data, entries[2].d->data, features, stream, false));
        }});
        methods.push_back({"V2C_BandAnchor", bandRows, [&]() {
            check(launchColumnFastOrdered(view, input.data, y.data, entries[3].d->data, features, stream, false));
        }});

        const std::uint64_t totalOcc = entries[0].st.totalRowOccurrences;
        const std::uint64_t atomicUpdates = totalOcc * features;

        std::cout << std::setprecision(9);
        std::cout << "dataset,F,method,band_rows,num_cw,num_tiles,num_atomic_updates,task_order_bytes,"
                     "preprocess_ms,upload_order_ms,kernel_ms,kernel_min,kernel_max,complete_ms,complete_min,"
                     "complete_max,speedup_vs_original_v2c,speedup_vs_identity_order,ratio_vs_acc,"
                     "acc_kernel_noclear_ms,relative_rmse,max_absolute_error\n" << std::flush;

        // First measure everything, then emit (speedup/ratio columns need the
        // original/identity/acc baselines, so no streaming per method).
        struct Measured { std::string name; U band; PairStats p; double noclear = 0; Error e; };
        std::vector<Measured> measured;
        for (auto& mt : methods) {
            PairStats p = measureColumn(mt.launch, y.data, yBytes, repeat, groups, launches, stream);
            double noclear = 0;
            if (std::string(mt.name) == "Acc_Basic") {
                Stats nc = measureGraph(mt.launch, repeat, groups, launches, stream); // no-clear kernel interval
                noclear = nc.med;
            }
            auto out = y.host(stream);
            Error e = error(head(out, real), head(accOut, real));
            require(e.rmse < 2e-4 && e.maxAbsolute < 2e-4 + e.maxReference * 5e-4,
                (std::string("vs-Acc mismatch for ") + mt.name).c_str());
            measured.push_back({mt.name, mt.bandRows, p, noclear, e});
        }
        auto kernelOf = [&](const char* name) -> float {
            for (auto& r : measured) if (r.name == name) return float(r.p.kernel.med);
            throw std::runtime_error(std::string("missing baseline ") + name);
        };
        const float tAcc = kernelOf("Acc_Basic"), tOrig = kernelOf("V2C_Original"), tIdent = kernelOf("V2C_IdentityOrder");
        for (auto& r : measured) {
            const float k = float(r.p.kernel.med);
            const double spOrig = k ? tOrig / k : 0;
            const double spIdent = k ? tIdent / k : 0;
            const double ratio = tAcc ? k / tAcc : 0;
            std::cout << argv[1] << ',' << features << ',' << r.name << ',' << r.band << ','
                      << W << ',' << base.tileLocalBit.size() << ',' << atomicUpdates << ','
                      << W * 4 << ',' << preprocessMs << ',' << uploadOrderMs << ','
                      << r.p.kernel.med << ',' << r.p.kernel.mn << ',' << r.p.kernel.mx << ','
                      << r.p.complete.med << ',' << r.p.complete.mn << ',' << r.p.complete.mx << ','
                      << spOrig << ',' << spIdent << ',' << ratio << ',' << r.noclear << ','
                      << r.e.rmse << ',' << r.e.maxAbsolute << '\n' << std::flush;
        }

        check(cudaStreamDestroy(stream));
    } catch (const std::exception& e) { std::cerr << "Error: " << e.what() << '\n'; return 1; }
    return 0;
}
