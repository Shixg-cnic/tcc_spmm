// INT4 fused-kernel benchmark: quantize X once (per-node scale, 2 nibbles/
// byte), then compare the fused INT4 kernels (fixed C0/C1/C3 and adaptive
// cache on/off) against the independent baseline (GPU dequantize + FP32 C1)
// and against the FP32 pipeline, on the same CSR/X/F/GPU/stream.
//
// Per method: correctness against a sampled CPU reference on the dequantised
// X-hat (TF32 math, what the kernels compute) and the quantization delta
// against a sampled CPU reference on the ORIGINAL X (reported, not asserted).
// Timing identical to benchmark_fp32: CUDA Graph + events, >=3 groups, median.
#include "matrix_io.cuh"
#include "column_schedule.cuh"
#include "int4_format.cuh"
#include <iomanip>

using namespace test;

namespace {

struct Stats { float med = 0, mn = 0, mx = 0; };
struct PairStats { Stats complete, kernel; };

Stats medMinMax(std::vector<float>& v) {
    std::sort(v.begin(), v.end());
    return {v[v.size()/2], v.front(), v.back()};
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
            float ms; check(cudaEventElapsedTime(&ms, begin, end));
            samples.push_back(ms / repeat);
        }
        cudaEventDestroy(begin); cudaEventDestroy(end);
        cudaGraphExecDestroy(exec); cudaGraphDestroy(graph);
    }
    return medMinMax(samples);
}

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
            float total; check(cudaEventElapsedTime(&total, begin, end));
            float kernel = 0;
            for (int i = 0; i < repeat; ++i) { float t; check(cudaEventElapsedTime(&t, start[i], stop[i])); kernel += t; }
            completes.push_back(total / repeat);
            kernels.push_back(kernel / repeat);
        }
        cudaEventDestroy(begin); cudaEventDestroy(end);
        cudaGraphExecDestroy(exec); cudaGraphDestroy(graph);
    }
    for (int i = 0; i < repeat; ++i) { cudaEventDestroy(start[i]); cudaEventDestroy(stop[i]); }
    return {medMinMax(completes), medMinMax(kernels)};
}

// CPU INT4 round-trip for the sampled references.
void quantizeCPU(const std::vector<float>& x, std::vector<std::uint8_t>& q,
                 std::vector<float>& scale, unsigned nodes, unsigned features) {
    const unsigned pairs = (features + 1) / 2;
    q.assign(std::size_t(nodes) * pairs, 0);
    scale.assign(nodes, 0.f);
    for (unsigned i = 0; i < nodes; ++i) {
        float m = 0.f;
        for (unsigned j = 0; j < features; ++j) m = std::max(m, std::fabs(x[std::size_t(i)*features+j]));
        const float s = m == 0.f ? 1.f : m / 7.f;
        scale[i] = s;
        for (unsigned j = 0; j < features; ++j) {
            int r = int(std::lround(x[std::size_t(i)*features+j] / s));
            r = std::max(-7, std::min(7, r));
            std::uint8_t nib = std::uint8_t(r) & 0xF;
            if (j % 2 == 0) q[std::size_t(i)*pairs + j/2] = nib;
            else q[std::size_t(i)*pairs + j/2] |= std::uint8_t(nib << 4);
        }
    }
}
std::vector<float> dequantCPU(const std::vector<std::uint8_t>& q, const std::vector<float>& scale,
                              unsigned nodes, unsigned features) {
    const unsigned pairs = (features + 1) / 2;
    std::vector<float> xh(std::size_t(nodes) * features, 0.f);
    for (unsigned i = 0; i < nodes; ++i)
        for (unsigned j = 0; j < features; ++j) {
            std::uint8_t nib = (j % 2 == 0) ? (q[std::size_t(i)*pairs + j/2] & 0xF)
                                            : (q[std::size_t(i)*pairs + j/2] >> 4);
            int qi = (nib & 8) ? int(nib) - 16 : int(nib);
            xh[std::size_t(i)*features+j] = float(qi) * scale[i];
        }
    return xh;
}

Error sampledCPU2(const Matrix& m, const std::vector<float>& x, const std::vector<float>& y,
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

const char* methodName(int method) {
    switch (method) {
        case 0: return "FP32-C1";
        case 1: return "FP32-C3";
        case 2: return "FP32-DynCache";
        case 3: return "FP32-DynNoCache";
        case 4: return "Int4-C0";
        case 5: return "Int4-C1";
        case 6: return "Int4-C3";
        case 7: return "Int4-DynCache";
        case 8: return "Int4-DynNoCache";
        case 9: return "Baseline-DeqC1";
        case 10: return "DequantOnly";
    }
    return "?";
}

// Schedule statistics (same feature set as benchmark_fp32 / model_calibration).
struct ScheduleStats {
    std::uint64_t tasks = 0, tiles = 0, occurrences = 0;
    std::uint64_t atomicBefore = 0, atomicAfter = 0;
    std::uint64_t maxTaskTiles = 0, meanTaskTiles100 = 0, p99TaskTiles = 0;
};
ScheduleStats scheduleStats(const ColumnBaseFormat& base, const ColumnSchedule& sched) {
    ScheduleStats s;
    s.tasks = sched.tasks.size();
    std::vector<std::uint64_t> perTask;
    for (auto& t : sched.tasks) {
        std::uint64_t tiles = 0;
        for (auto cw = t.cwBegin; cw < t.cwEnd; ++cw) {
            auto lo = std::max<std::uint64_t>(t.tileBegin, base.colWindowOffset[cw]);
            auto hi = std::min<std::uint64_t>(t.tileEnd, base.colWindowOffset[cw+1]);
            tiles += hi > lo ? hi - lo : 0;
        }
        perTask.push_back(tiles);
    }
    std::sort(perTask.begin(), perTask.end());
    for (auto v : perTask) s.tiles += v;
    for (std::uint64_t i = 0; i + 1 < base.colWindowOffset.size(); ++i) {
        std::uint64_t lo = base.colWindowOffset[i], hi = base.colWindowOffset[i+1];
        std::vector<std::uint32_t> rows;
        for (auto t = lo; t < hi; ++t)
            for (unsigned r = 0; r < 8; ++r)
                if (base.tileRows[std::size_t(t)*8+r] != UINT32_MAX) rows.push_back(base.tileRows[std::size_t(t)*8+r]);
        std::sort(rows.begin(), rows.end());
        rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
        s.occurrences += rows.size();
    }
    s.atomicBefore = s.occurrences * 128;
    s.atomicAfter = s.atomicBefore;
    for (std::uint32_t sw = 0; sw < sched.swCachedCount.size(); ++sw) {
        std::uint64_t saved = 0;
        if (sw < sched.tasks.size()) {
            auto& t = sched.tasks[sw];
            std::unordered_map<std::uint32_t, unsigned> freq;
            for (auto cw = t.cwBegin; cw < t.cwEnd; ++cw)
                for (auto tile = std::max<std::uint64_t>(t.tileBegin, base.colWindowOffset[cw]);
                     tile < std::min<std::uint64_t>(t.tileEnd, base.colWindowOffset[cw+1]); ++tile)
                    for (unsigned r = 0; r < 8; ++r) {
                        auto row = base.tileRows[std::size_t(tile)*8+r];
                        if (row != UINT32_MAX) ++freq[row];
                    }
            for (auto& [r, c] : freq) if (c > 1) saved += (c - 1);
        }
        s.atomicAfter -= saved * 128;
    }
    if (!perTask.empty()) {
        s.maxTaskTiles = perTask.back();
        s.meanTaskTiles100 = std::uint64_t(100.0 * double(s.tiles) / perTask.size());
        s.p99TaskTiles = perTask[std::size_t(0.99 * (perTask.size()-1))];
    }
    return s;
}

struct Int4Host {
    std::vector<std::uint8_t> q;
    std::vector<float> scale;
    std::vector<float> xhat;
};

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2 || std::string(argv[1]) == "--help") {
            std::cout << "Usage: benchmark_int4 matrix.csr [F=all|128] [groups=3] [device=0] [X.bin]\n"
                      << "all: 32,64,100,128,256,512. Quantises X once per F (per-node scale, INT4).\n";
            return 0;
        }
        int groups = argc > 3 ? std::stoi(argv[3]) : 3;
        require(groups >= 3, "at least 3 independent measurement groups required");
        check(cudaSetDevice(argc > 4 ? std::stoi(argv[4]) : 0));
        cudaStream_t stream; check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        auto m = readMatrix(argv[1]);
        const std::string dataset = std::string(argv[1]);
        auto t0 = Clock::now();
        auto col = convertColumn(m.rows, m.cols, m.csr);
        auto base = buildColumnBase(m.rows, m.cols, m.csr);
        auto sched = buildAdaptiveSchedule(base, ScheduleConfig{});
        auto fmt = buildAdaptiveColumn(base, sched);
        double convertMs = elapsed(t0);

        check(cudaDeviceSynchronize());
        t0 = Clock::now();
        GPUColumn gc(col);
        Buffer<std::uint32_t> win(fmt.colWindowOffset), slots(fmt.tileRowSlot), off(fmt.tileValueOffset),
            swOff(fmt.swRowOffset), rows(fmt.swRows), cacheSlots(fmt.swRowCacheSlot),
            cachedRows(fmt.swCachedRowSlot), counts(fmt.swCachedCount);
        Buffer<std::uint64_t> bits(fmt.tileLocalBit);
        Buffer<float> vals(fmt.values);
        Buffer<SWTask> tasks(fmt.tasks);
        DeviceColumnAdaptive view{fmt.rows, fmt.cols, std::uint32_t(fmt.tasks.size()),
            win.data, slots.data, off.data, bits.data, vals.data,
            swOff.data, rows.data, cacheSlots.data, cachedRows.data, counts.data, tasks.data};
        check(cudaDeviceSynchronize());
        double uploadMs = elapsed(t0);

        std::cout << std::setprecision(9);
        std::cout << "# benchmark_int4 dataset=" << dataset << " M=" << m.rows << " K=" << m.cols
                  << " nnz=" << m.csr.src.size() << " sw=" << col.superWindowSize
                  << " cache=" << COLUMN_CACHE_SIZE << " feature_tile=" << COLUMN_FEATURE_TILE
                  << " groups=" << groups << " tasks=" << fmt.tasks.size()
                  << " split_chunks=" << sched.splitChunks << '\n';
        {
            auto st = scheduleStats(base, sched);
            std::cout << "# dyn: tasks=" << st.tasks << " tiles=" << st.tiles
                      << " occurrences=" << st.occurrences << " atomic_before=" << st.atomicBefore
                      << " atomic_after=" << st.atomicAfter << " saved=" << (st.atomicBefore - st.atomicAfter)
                      << " max_task_tiles=" << st.maxTaskTiles << " mean=" << (st.meanTaskTiles100/100.0)
                      << " p99=" << st.p99TaskTiles << '\n';
        }
        std::cout << "dataset,M,K,nnz,F,method,kernel_ms,kernel_min,kernel_max,clear_ms,complete_ms,"
                     "complete_min,complete_max,rmse_vs_xhat,max_abs_vs_xhat,rmse_vs_x,max_abs_vs_x,"
                     "quantize_ms\n";

        std::vector<unsigned> dims{32, 64, 100, 128, 256, 512};
        if (argc > 2 && std::string(argv[2]) != "all") {
            int f = std::stoi(argv[2]); require(f > 0, "F must be positive"); dims = {unsigned(f)};
        }
        for (auto f : dims) {
            auto x = readFeatures(argc > 5 ? argv[5] : nullptr, m.cols, f);
            Int4Host h;
            quantizeCPU(x, h.q, h.scale, m.cols, f);
            h.xhat = dequantCPU(h.q, h.scale, m.cols, f);
            Buffer<float> input(x);
            Buffer<std::uint8_t> xq(h.q);
            Buffer<float> scaleG(h.scale);
            Buffer<float> xhatG(h.xhat);
            DeviceInt4X x4{xq.data, scaleG.data};
            Buffer<float> y(std::size_t(m.rows) * f);
            const std::size_t yBytes = y.size * sizeof(float);
            // quantize timing (one-time per F).
            float quantMs = 0;
            {
                auto launch = [&]() { check(quantizeXInt4(input.data, xq.data, scaleG.data, m.cols, f, stream)); };
                quantMs = measureGraph(launch, 1, groups, 3, stream).med;
            }
            std::cout << "# dataset=" << dataset << " F=" << f << " quantize_ms=" << quantMs << '\n';

            for (int method = 0; method < 11; ++method) {
                const bool dequantOnly = method == 10;
                const bool baseline = method == 9;
                const bool fp32 = method <= 3;
                // correctness run
                if (dequantOnly) {
                    check(cudaMemsetAsync(y.data, 0, yBytes, stream));
                } else if (baseline) {
                    check(cudaMemsetAsync(y.data, 0, yBytes, stream));
                    check(dequantizeXInt4(xq.data, scaleG.data, xhatG.data, m.cols, f, stream));
                    check(launchColumnSpMM(gc.view, xhatG.data, y.data, f, ColumnVersion::C1, stream, false));
                } else if (fp32) {
                    check(cudaMemsetAsync(y.data, 0, yBytes, stream));
                    if (method == 0) check(launchColumnSpMM(gc.view, input.data, y.data, f, ColumnVersion::C1, stream, false));
                    else if (method == 1) check(launchColumnSpMM(gc.view, input.data, y.data, f, ColumnVersion::C3, stream, false));
                    else if (method == 2) check(launchColumnAdaptiveSpMM(view, input.data, y.data, f, true, stream, false));
                    else check(launchColumnAdaptiveSpMM(view, input.data, y.data, f, false, stream, false));
                } else {
                    check(cudaMemsetAsync(y.data, 0, yBytes, stream));
                    if (method == 4) check(launchColumnInt4SpMM(gc.view, x4, y.data, f, ColumnVersion::C0, stream, false));
                    else if (method == 5) check(launchColumnInt4SpMM(gc.view, x4, y.data, f, ColumnVersion::C1, stream, false));
                    else if (method == 6) check(launchColumnInt4SpMM(gc.view, x4, y.data, f, ColumnVersion::C3, stream, false));
                    else if (method == 7) check(launchColumnAdaptiveInt4SpMM(view, x4, y.data, f, true, stream, false));
                    else check(launchColumnAdaptiveInt4SpMM(view, x4, y.data, f, false, stream, false));
                }
                auto yHost = y.host(stream);
                std::vector<float> valid(yHost.begin(), yHost.begin() + std::size_t(m.rows) * f);
                auto eXhat = sampledCPU2(m, h.xhat, valid, f, true);
                auto eX = (dequantOnly || baseline || fp32) ? Error{0,0,0} : sampledCPU2(m, x, valid, f, false);
                // timing
                const int repeat = 5;
                PairStats pair; Stats clearStat;
                if (dequantOnly) {
                    auto launch = [&]() { check(dequantizeXInt4(xq.data, scaleG.data, xhatG.data, m.cols, f, stream)); };
                    auto cs = measureColumn(launch, y.data, yBytes, repeat, groups, 3, stream);
                    pair = cs;
                    auto clear = [&]() { check(cudaMemsetAsync(y.data, 0, yBytes, stream)); };
                    clearStat = measureGraph(clear, repeat, groups, 3, stream);
                } else if (baseline) {
                    auto launch = [&]() {
                        check(dequantizeXInt4(xq.data, scaleG.data, xhatG.data, m.cols, f, stream));
                        check(launchColumnSpMM(gc.view, xhatG.data, y.data, f, ColumnVersion::C1, stream, false));
                    };
                    auto cs = measureColumn(launch, y.data, yBytes, repeat, groups, 3, stream);
                    pair = cs;
                    auto clear = [&]() { check(cudaMemsetAsync(y.data, 0, yBytes, stream)); };
                    clearStat = measureGraph(clear, repeat, groups, 3, stream);
                } else if (fp32) {
                    auto launch = [&]() {
                        if (method == 0) check(launchColumnSpMM(gc.view, input.data, y.data, f, ColumnVersion::C1, stream, false));
                        else if (method == 1) check(launchColumnSpMM(gc.view, input.data, y.data, f, ColumnVersion::C3, stream, false));
                        else if (method == 2) check(launchColumnAdaptiveSpMM(view, input.data, y.data, f, true, stream, false));
                        else check(launchColumnAdaptiveSpMM(view, input.data, y.data, f, false, stream, false));
                    };
                    auto cs = measureColumn(launch, y.data, yBytes, repeat, groups, 3, stream);
                    pair = cs;
                    auto clear = [&]() { check(cudaMemsetAsync(y.data, 0, yBytes, stream)); };
                    clearStat = measureGraph(clear, repeat, groups, 3, stream);
                } else {
                    auto launch = [&]() {
                        if (method == 4) check(launchColumnInt4SpMM(gc.view, x4, y.data, f, ColumnVersion::C0, stream, false));
                        else if (method == 5) check(launchColumnInt4SpMM(gc.view, x4, y.data, f, ColumnVersion::C1, stream, false));
                        else if (method == 6) check(launchColumnInt4SpMM(gc.view, x4, y.data, f, ColumnVersion::C3, stream, false));
                        else if (method == 7) check(launchColumnAdaptiveInt4SpMM(view, x4, y.data, f, true, stream, false));
                        else check(launchColumnAdaptiveInt4SpMM(view, x4, y.data, f, false, stream, false));
                    };
                    auto cs = measureColumn(launch, y.data, yBytes, repeat, groups, 3, stream);
                    pair = cs;
                    auto clear = [&]() { check(cudaMemsetAsync(y.data, 0, yBytes, stream)); };
                    clearStat = measureGraph(clear, repeat, groups, 3, stream);
                }
                std::cout << dataset << ',' << m.rows << ',' << m.cols << ',' << m.csr.src.size() << ',' << f << ','
                          << methodName(method) << ',' << pair.kernel.med << ',' << pair.kernel.mn << ','
                          << pair.kernel.mx << ',' << clearStat.med << ',' << pair.complete.med << ','
                          << pair.complete.mn << ',' << pair.complete.mx << ',' << eXhat.rmse << ','
                          << eXhat.maxAbsolute << ',' << eX.rmse << ',' << eX.maxAbsolute << ',' << quantMs
                          << '\n' << std::flush;
            }
        }
        check(cudaStreamDestroy(stream));
    } catch (const std::exception& e) { std::cerr << "Error: " << e.what() << '\n'; return 1; }
    return 0;
}
