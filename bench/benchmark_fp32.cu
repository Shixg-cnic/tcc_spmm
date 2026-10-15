// Unified FP32 baseline benchmark: original Acc tf32_mma_kernel vs the local
// Row kernel and Column C0/C1/C2/C3, on the same CSR / X / F / GPU / stream.
//
// Timing uses CUDA Graph + Events with explicit warmup and >=3 independent
// capture groups; the median and the min..max spread are reported.
// Column "complete" always includes the required Y memset; the isolated kernel
// interval is measured inside the same graph so clock state is identical.
//
// Acc constraints (recorded, original source untouched):
//   - features must be in [64,512] and a multiple of 32 (F=32/100 skipped);
//   - rows are padded to a multiple of 8 (padded output rows are discarded);
//   - empty 8-row windows are unsupported (verified absent on the target graphs).
#include "matrix_io.cuh"
#include "acc_reference.cuh"
#include "column_schedule.cuh"
#include <iomanip>
#include <memory>

using namespace test;

namespace {

struct Stats { float med = 0, mn = 0, mx = 0; };
struct PairStats { Stats complete, kernel; };

std::vector<float> samplesOf;

Stats medMinMax(std::vector<float>& v) {
    std::sort(v.begin(), v.end());
    return {v[v.size()/2], v.front(), v.back()};
}

// Capture 'repeat' capture-safe operations, instantiate, warm up, then time
// 'launches' graph launches per group; returns per-iteration statistics.
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

// Column graph: per iteration memsetAsync(Y) then external events around the
// kernel. Complete and isolated-kernel intervals come from the SAME launches,
// so their clock state is identical (avoids the kernel>complete anomaly seen
// when the two were measured in separate sequences).
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

std::size_t rowBytes(const RowBitTCF& m) {
    return 4 * (m.rowWindowOffset.size() + m.tcOffset.size() + m.sparseA2B.size() + m.values.size()) +
           8 * m.tcLocalBit.size();
}
std::size_t columnBytes(const ColumnSWTCF& m) {
    return 4 * (m.colWindowOffset.size() + m.tileRowSlot.size() + m.tileValueOffset.size() + m.values.size() +
                m.superWindowRowOffset.size() + m.superWindowRows.size() + m.superWindowRowCacheSlot.size() +
                m.superWindowCachedRowSlot.size() + m.superWindowCachedCount.size()) +
           8 * m.tileLocalBit.size();
}
// Base column format without the SW output-cache metadata (only C3 uses it):
// tile data + SW row mapping, no cache-slot/cache-dir arrays.
std::size_t columnBytesBase(const ColumnSWTCF& m) {
    return 4 * (m.colWindowOffset.size() + m.tileRowSlot.size() + m.tileValueOffset.size() + m.values.size() +
                m.superWindowRowOffset.size() + m.superWindowRows.size()) +
           8 * m.tileLocalBit.size();
}

struct Formats {
    RowBitTCF row;
    RowBitTCF acc;          // original Acc METCFBit, verified equal to row
    ColumnSWTCF column;
    ColumnBaseFormat columnBase;
    ColumnSchedule schedDyn, schedNoSplit;
    ColumnAdaptiveFormat adaptiveDyn, adaptiveNoSplit;
    std::size_t columnBaseBytes = 0;
    double rowConvertMs, accConvertMs, colConvertMs, adaptiveConvertMs;
    std::size_t rowBytes, accBytes, columnBytes, adaptiveBytes;
};

std::size_t adaptiveBytes(const ColumnAdaptiveFormat& m) {
    return 4 * (m.colWindowOffset.size() + m.tileRowSlot.size() + m.tileValueOffset.size() + m.values.size() +
                m.swRowOffset.size() + m.swRows.size() + m.swRowCacheSlot.size() +
                m.swCachedRowSlot.size() + m.swCachedCount.size() + m.tasks.size()) +
           8 * m.tileLocalBit.size();
}

Formats buildFormats(const Matrix& m, unsigned swSize) {
    Formats f;
    auto t0 = Clock::now(); f.row = convertRow(m.rows, m.cols, m.csr); f.rowConvertMs = elapsed(t0);
    t0 = Clock::now(); f.acc = convertAccRow(m.rows, m.cols, m.csr); f.accConvertMs = elapsed(t0);
    t0 = Clock::now(); f.column = convertColumn(m.rows, m.cols, m.csr, swSize); f.colConvertMs = elapsed(t0);
    if (f.acc.rowWindowOffset != f.row.rowWindowOffset || f.acc.tcOffset != f.row.tcOffset ||
        f.acc.sparseA2B != f.row.sparseA2B || f.acc.tcLocalBit != f.row.tcLocalBit ||
        f.acc.values != f.row.values)
        throw std::runtime_error("original Acc format differs from local Row format");
    f.rowBytes = rowBytes(f.row); f.accBytes = f.rowBytes; f.columnBytes = columnBytes(f.column);
    f.columnBaseBytes = columnBytesBase(f.column);
    t0 = Clock::now();
    f.columnBase = buildColumnBase(m.rows, m.cols, m.csr);
    f.schedDyn = buildAdaptiveSchedule(f.columnBase, ScheduleConfig{});
    f.adaptiveDyn = buildAdaptiveColumn(f.columnBase, f.schedDyn);
    ScheduleConfig noSplit;
    noSplit.splitTileThreshold = UINT32_MAX;
    f.schedNoSplit = buildAdaptiveSchedule(f.columnBase, noSplit);
    f.adaptiveNoSplit = buildAdaptiveColumn(f.columnBase, f.schedNoSplit);
    f.adaptiveConvertMs = elapsed(t0);
    f.adaptiveBytes = adaptiveBytes(f.adaptiveDyn);
    return f;
}

// Uploaded adaptive formats (both schedules) with their GPU views.
struct GPUAdaptivePair {
    Buffer<std::uint32_t> dynWindows, dynSlots, dynOffset, dynSwOffset, dynRows, dynCacheSlots, dynCachedRows, dynCounts;
    Buffer<std::uint64_t> dynBits;
    Buffer<float> dynValues;
    Buffer<SWTask> dynTasks;
    Buffer<std::uint32_t> nsWindows, nsSlots, nsOffset, nsSwOffset, nsRows, nsCacheSlots, nsCachedRows, nsCounts;
    Buffer<std::uint64_t> nsBits;
    Buffer<float> nsValues;
    Buffer<SWTask> nsTasks;
    DeviceColumnAdaptive viewDyn, viewNoSplit;
    explicit GPUAdaptivePair(const Formats& f)
        : dynWindows(f.adaptiveDyn.colWindowOffset), dynSlots(f.adaptiveDyn.tileRowSlot),
          dynOffset(f.adaptiveDyn.tileValueOffset), dynSwOffset(f.adaptiveDyn.swRowOffset),
          dynRows(f.adaptiveDyn.swRows), dynCacheSlots(f.adaptiveDyn.swRowCacheSlot),
          dynCachedRows(f.adaptiveDyn.swCachedRowSlot), dynCounts(f.adaptiveDyn.swCachedCount),
          dynBits(f.adaptiveDyn.tileLocalBit), dynValues(f.adaptiveDyn.values), dynTasks(f.adaptiveDyn.tasks),
          nsWindows(f.adaptiveNoSplit.colWindowOffset), nsSlots(f.adaptiveNoSplit.tileRowSlot),
          nsOffset(f.adaptiveNoSplit.tileValueOffset), nsSwOffset(f.adaptiveNoSplit.swRowOffset),
          nsRows(f.adaptiveNoSplit.swRows), nsCacheSlots(f.adaptiveNoSplit.swRowCacheSlot),
          nsCachedRows(f.adaptiveNoSplit.swCachedRowSlot), nsCounts(f.adaptiveNoSplit.swCachedCount),
          nsBits(f.adaptiveNoSplit.tileLocalBit), nsValues(f.adaptiveNoSplit.values), nsTasks(f.adaptiveNoSplit.tasks),
          viewDyn{f.adaptiveDyn.rows, f.adaptiveDyn.cols, uint32_t(f.adaptiveDyn.tasks.size()),
              dynWindows.data, dynSlots.data, dynOffset.data, dynBits.data, dynValues.data,
              dynSwOffset.data, dynRows.data, dynCacheSlots.data, dynCachedRows.data, dynCounts.data, dynTasks.data},
          viewNoSplit{f.adaptiveNoSplit.rows, f.adaptiveNoSplit.cols, uint32_t(f.adaptiveNoSplit.tasks.size()),
              nsWindows.data, nsSlots.data, nsOffset.data, nsBits.data, nsValues.data,
              nsSwOffset.data, nsRows.data, nsCacheSlots.data, nsCachedRows.data, nsCounts.data, nsTasks.data} {}
};

const char* methodName(int method) {
    switch (method) {
        case 0: return "Acc-Original";
        case 1: return "Row";
        case 2: return "C0";
        case 3: return "C1";
        case 4: return "C2";
        case 5: return "C3";
        case 6: return "DynCache";
        case 7: return "DynNoSplit";
        case 8: return "DynNoCache";
    }
    return "?";
}

// Per-method (format-bytes, conversion-ms, upload) helper for the CSV row.
struct MethodMeta { std::size_t bytes; double convertMs; };

// Aggregate schedule statistics used by the calibrated cost model (3.6):
// tiles, (CW,row) occurrences, atomics before/after the SW cache, per-task
// tile distribution.
struct ScheduleStats {
    std::uint64_t tasks = 0, tiles = 0, occurrences = 0;
    std::uint64_t atomicBefore = 0, atomicAfter = 0;
    std::uint64_t maxTaskTiles = 0, meanTaskTiles100 = 0, p99TaskTiles = 0;
};
ScheduleStats scheduleStats(const ColumnBaseFormat& base, const ColumnSchedule& sched, unsigned f) {
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
    s.atomicBefore = s.occurrences * f;
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
        s.atomicAfter -= saved * f;
    }
    if (!perTask.empty()) {
        s.maxTaskTiles = perTask.back();
        s.meanTaskTiles100 = std::uint64_t(100.0 * double(s.tiles) / perTask.size());
        s.p99TaskTiles = perTask[std::size_t(0.99 * (perTask.size()-1))];
    }
    return s;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2 || std::string(argv[1]) == "--help") {
            std::cout << "Usage: benchmark_fp32 matrix.csr [F=all|128] [groups=3] [device=0] [X.bin] [SW=16]\n"
                      << "all: 32,64,100,128,256,512. X.bin is K*sourceF raw FP32; truncate/cyclic-expand.\n"
                      << "SW: Column SuperWindow size used for conversion (default 16).\n"
                      << "Output: one CSV row per (dataset,F,method) with median and min..max spread.\n";
            return 0;
        }
        int groups = argc > 3 ? std::stoi(argv[3]) : 3;
        require(groups >= 3, "at least 3 independent measurement groups required");
        check(cudaSetDevice(argc > 4 ? std::stoi(argv[4]) : 0));
        unsigned swSize = argc > 6 ? unsigned(std::stoi(argv[6])) : COLUMN_SW_SIZE;
        require(swSize > 0, "SW must be positive");
        cudaStream_t stream; check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        auto m = readMatrix(argv[1]);
        const std::string dataset = std::string(argv[1]);
        auto formats = buildFormats(m, swSize);
        const std::uint32_t paddedRows = (std::uint64_t(m.rows) + 7) / 8 * 8;

        check(cudaDeviceSynchronize());
        auto t0 = Clock::now();
        GPUColumn gc(formats.column); GPURow gr(formats.row);
        GPUAdaptivePair gp(formats);
        Buffer<std::uint64_t> accBits(formats.acc.tcLocalBit);
        Buffer<std::uint32_t> accWindow(formats.acc.rowWindowOffset), accOffset(formats.acc.tcOffset),
            accMapping(formats.acc.sparseA2B);
        Buffer<float> accValues(formats.acc.values);
        check(cudaDeviceSynchronize());
        double uploadMs = elapsed(t0);

        std::cout << std::setprecision(9);
        std::cout << "# benchmark_fp32 dataset=" << dataset << " M=" << m.rows << " K=" << m.cols
                  << " nnz=" << m.csr.src.size() << " sw=" << formats.column.superWindowSize
                  << " cache=" << COLUMN_CACHE_SIZE << " feature_tile=" << COLUMN_FEATURE_TILE
                  << " groups=" << groups << " padded_rows=" << paddedRows << '\n';
        std::cout << "# Acc constraints: F in [64,512] and F%32==0; rows padded to 8; no empty 8-row windows.\n";
        for (int v = 0; v < 2; ++v) {
            auto st = scheduleStats(formats.columnBase, v ? formats.schedNoSplit : formats.schedDyn, 128);
            std::cout << "# dyn" << (v ? "-nosplit" : "") << ": tasks=" << st.tasks << " tiles=" << st.tiles
                      << " occurrences=" << st.occurrences << " atomic_before=" << st.atomicBefore
                      << " atomic_after=" << st.atomicAfter << " saved=" << (st.atomicBefore - st.atomicAfter)
                      << " max_task_tiles=" << st.maxTaskTiles << " mean=" << (st.meanTaskTiles100/100.0)
                      << " p99=" << st.p99TaskTiles << '\n';
        }
        std::cout << "# dynamic schedule: tasks_dyn=" << formats.adaptiveDyn.tasks.size()
                  << " split_chunks_dyn=" << formats.schedDyn.splitChunks
                  << " tasks_nosplit=" << formats.adaptiveNoSplit.tasks.size() << '\n';
        std::cout << "dataset,M,K,nnz,F,method,format_bytes,format_conversion_ms,upload_ms,kernel_ms,"
                     "kernel_min,kernel_max,output_clear_ms,complete_spmm_ms,complete_min,complete_max,"
                     "relative_rmse,max_absolute_error,rmse_vs_row,max_abs_vs_row\n";

        std::vector<unsigned> dims{32, 64, 100, 128, 256, 512};
        if (argc > 2 && std::string(argv[2]) != "all") {
            int f = std::stoi(argv[2]); require(f > 0, "F must be positive"); dims = {unsigned(f)};
        }
        for (auto f : dims) {
            auto x = readFeatures(argc > 5 ? argv[5] : nullptr, m.cols, f);
            Buffer<float> input(x);
            Buffer<float> rowY(std::size_t(m.rows) * f);
            gr.launch(input.data, rowY.data, f, stream);
            auto rowOut = rowY.host(stream);
            std::cout << "# dataset=" << dataset << " M=" << m.rows << " K=" << m.cols << " nnz=" << m.csr.src.size()
                      << " F=" << f << '\n';

            for (int method = 0; method < 9; ++method) {
                const bool acc = method == 0;
                if (acc && (f < 64 || f % 32)) {
                    std::cout << "# Acc-Original skipped for F=" << f << " (constraint F in [64,512], F%32==0)\n";
                    continue;
                }
                const bool rowMethod = method == 1;
                const bool columnMethod = method >= 2 && method <= 5;
                const bool adaptiveMethod = method >= 6;
                const DeviceColumnAdaptive* dynView = method == 6 || method == 8 ? &gp.viewDyn : &gp.viewNoSplit;
                const bool dynCache = method == 6 || method == 7;
                std::size_t yCount = (acc ? std::size_t(paddedRows) : std::size_t(m.rows)) * f;
                Buffer<float> y(yCount);
                // --- correctness run ---
                if (acc) {
                    check(launchAccRowSpMM(accWindow.data, accOffset.data, accMapping.data, accBits.data,
                        accValues.data, input.data, y.data, paddedRows, m.cols, f, stream));
                } else if (rowMethod) {
                    gr.launch(input.data, y.data, f, stream);
                } else if (columnMethod) {
                    check(launchColumnSpMM(gc.view, input.data, y.data, f, ColumnVersion(method - 2), stream));
                } else {
                    check(launchColumnAdaptiveSpMM(*dynView, input.data, y.data, f, dynCache, stream));
                }
                auto yHost = y.host(stream);
                std::vector<float> valid(yHost.begin(), yHost.begin() + std::size_t(m.rows) * f);
                auto cpu = sampledCPU(m, x, valid, f, true);
                auto vsRow = error(valid, rowOut);
                // --- timing ---
                const int repeat = 5;
                float kernelMs, clearMs = 0, completeMs;
                Stats kstat, cstat, clearStat;
                if (acc || rowMethod) {
                    auto launch = [&]() {
                        if (acc) check(launchAccRowSpMM(accWindow.data, accOffset.data, accMapping.data,
                            accBits.data, accValues.data, input.data, y.data, paddedRows, m.cols, f, stream));
                        else gr.launch(input.data, y.data, f, stream);
                    };
                    cstat = measureGraph(launch, repeat, groups, 3, stream);
                    kernelMs = cstat.med; completeMs = cstat.med;
                    kstat = cstat;
                } else if (columnMethod) {
                    auto launch = [&]() { check(launchColumnSpMM(gc.view, input.data, y.data, f,
                        ColumnVersion(method - 2), stream, false)); };
                    auto pair = measureColumn(launch, y.data, yCount * sizeof(float), repeat, groups, 3, stream);
                    kstat = pair.kernel; cstat = pair.complete;
                    auto clear = [&]() { check(cudaMemsetAsync(y.data, 0, yCount * sizeof(float), stream)); };
                    clearStat = measureGraph(clear, repeat, groups, 3, stream);
                    kernelMs = kstat.med; clearMs = clearStat.med; completeMs = cstat.med;
                } else {
                    auto launch = [&]() { check(launchColumnAdaptiveSpMM(*dynView, input.data, y.data, f,
                        dynCache, stream, false)); };
                    auto pair = measureColumn(launch, y.data, yCount * sizeof(float), repeat, groups, 3, stream);
                    kstat = pair.kernel; cstat = pair.complete;
                    auto clear = [&]() { check(cudaMemsetAsync(y.data, 0, yCount * sizeof(float), stream)); };
                    clearStat = measureGraph(clear, repeat, groups, 3, stream);
                    kernelMs = kstat.med; clearMs = clearStat.med; completeMs = cstat.med;
                }
                MethodMeta meta;
                if (acc) meta = {formats.accBytes, formats.accConvertMs};
                else if (rowMethod) meta = {formats.rowBytes, formats.rowConvertMs};
                else if (adaptiveMethod) meta = {formats.adaptiveBytes, formats.adaptiveConvertMs};
                else meta = {method == 5 ? formats.columnBytes : formats.columnBaseBytes, formats.colConvertMs};
                std::cout << dataset << ',' << m.rows << ',' << m.cols << ',' << m.csr.src.size() << ',' << f << ','
                          << methodName(method) << ',' << meta.bytes << ',' << meta.convertMs << ',' << uploadMs << ','
                          << kernelMs << ',' << kstat.mn << ',' << kstat.mx << ',' << clearMs << ',' << completeMs << ','
                          << cstat.mn << ',' << cstat.mx << ',' << cpu.rmse << ',' << cpu.maxAbsolute << ','
                          << vsRow.rmse << ',' << vsRow.maxAbsolute << '\n' << std::flush;
            }
        }
        check(cudaStreamDestroy(stream));
    } catch (const std::exception& e) { std::cerr << "Error: " << e.what() << '\n'; return 1; }
    return 0;
}
