// Output-path diagnosis for the Column kernels (task-book batch 1).
//
// D0  : full current C0 kernel (real reference, correct result, verified).
// D1  : full C0 compute (sparse load, X load, TF32 MMA, output row mapping,
//       Y address computation) with NO global writeback; a per-CTA checksum
//       keeps every address/value dependency alive so the compiler cannot
//       delete the address chain. NOT a correct SpMM, timing only.
// D1b : like D1 but issues plain (non-atomic) scattered stores to Y at the
//       same addresses -- separates store-instruction cost from the atomic
//       RMW serialization. NOT a correct SpMM.
// D2  : atomic-only: identical CTA schedule, CW/tile iteration and output
//       address stream as C0, no X load and no MMA; every (tile,row,feature)
//       does one atomicAdd of a constant. Measures the effective atomic
//       throughput under the real address distribution. NOT a correct SpMM.
// D3  : offline analysis of the column format: per-row update counts and
//       per-row distinct-CW counts (contention), P50/P95/P99/max, CTA-per-row,
//       CW-index spread per row, total logical atomics, Y working set.
#include "matrix_io.cuh"
#include "column_format.cuh"
#include "column_schedule.cuh"
#include "mma_tf32.cuh"
#include <iomanip>

using namespace test;
namespace {
using U = std::uint32_t;

// ---------------------------------------------------------------------------
// Diagnostic kernels. Mode: 0 = D1 no-writeback checksum,
//                          1 = D1b non-atomic scattered store,
//                          2 = D2 atomic-only (no MMA).
// D0 uses the real launchColumnSpMM(C0).
// ---------------------------------------------------------------------------
__device__ void diagLoadX(const float* x, U w, U cols, U f, U features, float b[2][4]) {
    U lane = threadIdx.x % 32, group = lane / 4, k = lane % 4;
    #pragma unroll
    for (int part = 0; part < 2; ++part)
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            U col = w * 8 + k + (i >= 2 ? 4 : 0);
            U feature = f + group + (i % 2) * 8 + part * 16;
            b[part][i] = col < cols && feature < features ? x[std::size_t(col) * features + feature] : 0.f;
        }
}

template<int Mode>
__global__ void diagColumnKernel(DeviceColumnSWTCF m, const float* x, float* y,
    float* sink, U features) {
    __shared__ float a[64];
    __shared__ U slots[8];
    const U windows = (std::uint64_t(m.cols) + 7) / 8;
    const U w = blockIdx.x;
    const U sw = w / m.superWindowSize;
    const U rowBase = m.superWindowRowOffset[sw];
    const U blockFeature = blockIdx.y * COLUMN_FEATURE_TILE;
    const U fBase = blockFeature + (threadIdx.x / 32) * 32;
    const U lane = threadIdx.x % 32, group = lane / 4, k = lane % 4;

    float b[2][4];
    if constexpr (Mode != 2 && Mode != 3) diagLoadX(x, w, m.cols, fBase, features, b);
    float checksum = 0.f;

    const U first = m.colWindowOffset[w], end = m.colWindowOffset[w + 1];
    for (U tile = first; tile < end; ++tile) {
        if constexpr (Mode != 2 && Mode != 3) {
            if (threadIdx.x < 64) {
                auto bitmap = m.tileLocalBit[tile];
                a[threadIdx.x] = (bitmap & (1ULL << threadIdx.x)) ?
                    m.values[m.tileValueOffset[tile] + __popcll(bitmap & ((1ULL << threadIdx.x) - 1))] : 0.f;
            }
            if (threadIdx.x < 8) slots[threadIdx.x] = m.tileRowSlot[std::size_t(tile) * 8 + threadIdx.x];
            __syncthreads();
            float sparse[2] = {a[group * 8 + k], a[group * 8 + k + 4]};
            #pragma unroll
            for (int part = 0; part < 2; ++part) {
                float c[4] = {0.f, 0.f, 0.f, 0.f};
                tf32_m16n8k8(b[part], sparse, c);
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    U slot = slots[k * 2 + i % 2];
                    U feature = fBase + group + (i >= 2 ? 8 : 0) + part * 16;
                    if (slot != UINT32_MAX && feature < features) {
                        U row = m.superWindowRows[rowBase + slot];
                        std::size_t addr = std::size_t(row) * features + feature;
                        if constexpr (Mode == 0) checksum += c[i] + float(addr & 0xFF); // keep addr chain alive
                        else y[addr] = c[i];                                             // D1b: plain store
                    }
                }
            }
        } else {
            // D2 (Mode 2): REDG STRONG (what C0 emits); D2b (Mode 3): WEAK scope.
            #pragma unroll
            for (int part = 0; part < 2; ++part)
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    U slot = m.tileRowSlot[std::size_t(tile) * 8 + k * 2 + i % 2];
                    U feature = fBase + group + (i >= 2 ? 8 : 0) + part * 16;
                    if (slot != UINT32_MAX && feature < features) {
                        U row = m.superWindowRows[rowBase + slot];
                        float* p = y + std::size_t(row) * features + feature;
                        atomicAdd(p, 1.f); // compiler -> REDG.E.ADD.F32.STRONG.GPU (as in C0)
                    }
                }
            __syncthreads();
        }
    }
    if constexpr (Mode == 0) {
        // One observable write per CTA keeps the whole address/value chain live.
        sink[std::size_t(blockIdx.x) * gridDim.y + blockIdx.y] = checksum;
    }
}

// ---------------------------------------------------------------------------
// D3 offline analysis.
// ---------------------------------------------------------------------------
struct D3Stats {
    std::uint64_t tiles = 0, cwPairs = 0, atomics = 0;
    std::uint64_t rows = 0;
    double p50 = 0, p95 = 0, p99 = 0, maxCw = 0, maxTile = 0;
    double meanCw = 0, meanTile = 0;
    double meanSpread = 0, maxSpread = 0;
    double yWorkingSetMB = 0;
    std::uint64_t rowsContended2 = 0, rowsContended16 = 0;
};
D3Stats analyzeD3(const ColumnBaseFormat& base, U features) {
    D3Stats s;
    const auto& off = base.colWindowOffset;
    const U windows = U(off.size() - 1);
    std::vector<U> rowTile(0), rowCwMin(base.rows, UINT32_MAX), rowCwMax(base.rows, 0);
    std::vector<std::uint64_t> rowTileCount(base.rows, 0), rowCwCount(base.rows, 0);
    std::vector<U> lastCw(base.rows, UINT32_MAX);
    for (U w = 0; w < windows; ++w) {
        for (std::uint64_t t = off[w]; t < off[w + 1]; ++t) {
            ++s.tiles;
            for (unsigned r = 0; r < 8; ++r) {
                U row = base.tileRows[t * 8 + r];
                if (row == UINT32_MAX) continue;
                ++rowTileCount[row];
                if (lastCw[row] != w) { ++rowCwCount[row]; lastCw[row] = w; }
                rowCwMin[row] = std::min(rowCwMin[row], w);
                rowCwMax[row] = std::max(rowCwMax[row], w);
            }
        }
    }
    std::vector<std::uint64_t> tileDist, cwDist, spreadDist;
    tileDist.reserve(base.rows); cwDist.reserve(base.rows); spreadDist.reserve(base.rows);
    for (U r = 0; r < base.rows; ++r) {
        if (!rowTileCount[r]) continue;
        ++s.rows;
        s.atomics += rowTileCount[r] * features;
        tileDist.push_back(rowTileCount[r]);
        cwDist.push_back(rowCwCount[r]);
        std::uint64_t spread = rowCwMax[r] >= rowCwMin[r] ? rowCwMax[r] - rowCwMin[r] : 0;
        spreadDist.push_back(spread);
    }
    auto pct = [](std::vector<std::uint64_t>& v, double p) {
        if (v.empty()) return 0.0;
        std::sort(v.begin(), v.end());
        return double(v[std::size_t(p * (v.size() - 1))]);
    };
    auto mean = [](std::vector<std::uint64_t>& v) {
        if (v.empty()) return 0.0;
        return double(std::accumulate(v.begin(), v.end(), std::uint64_t(0))) / v.size();
    };
    s.p50 = pct(cwDist, 0.50); s.p95 = pct(cwDist, 0.95); s.p99 = pct(cwDist, 0.99);
    s.maxCw = pct(cwDist, 1.0); s.maxTile = pct(tileDist, 1.0);
    s.meanCw = mean(cwDist); s.meanTile = mean(tileDist);
    s.meanSpread = mean(spreadDist); s.maxSpread = pct(spreadDist, 1.0);
    s.yWorkingSetMB = double(base.rows) * features * 4.0 / (1024.0 * 1024.0);
    for (U r = 0; r < base.rows; ++r) {
        if (rowCwCount[r] >= 2) ++s.rowsContended2;
        if (rowCwCount[r] >= 16) ++s.rowsContended16;
    }
    return s;
}

struct Stats { float med = 0, mn = 0, mx = 0; };

// D2c: same REDG COUNT as the real kernel but all on a small L2-resident
// contiguous scratch buffer (no DRAM RMW, no address competition).
// Distinguishes hardware atomic-pipe throughput from address-competition/DRAM.
// Address pattern: thread t of a warp hits consecutive floats -> one 128B line
// per warp-instruction (dense L2 bursts); successive iterations stride 128.
__global__ void diagAtomicContiguous(U opsPerThread, float* scratch) {
    const U tid = threadIdx.x;
    U i = 0;
    #pragma unroll 8
    for (; i + 8 <= opsPerThread; i += 8) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) atomicAdd(scratch + ((tid + (i + j) * 128u) & 0x3FFFFF), 1.f);
    }
    for (; i < opsPerThread; ++i) atomicAdd(scratch + ((tid + i * 128u) & 0x3FFFFF), 1.f);
}
Stats medMinMax(std::vector<float>& v) {
    std::sort(v.begin(), v.end());
    return {v[v.size() / 2], v.front(), v.back()};
}
Stats measureGraph(const std::function<void()>& op, int repeat, int groups, int launches,
    cudaStream_t stream) {
    std::vector<float> samples;
    for (int g = 0; g < groups; ++g) {
        cudaGraph_t graph; cudaGraphExec_t exec;
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        for (int i = 0; i < repeat; ++i) op();
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
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2 || std::string(argv[1]) == "--help") {
            std::cout << "Usage: column_output_diagnosis matrix.csr [F=128|256] [groups=3] [device=0] [X.bin] [SW=16]\n"
                      << "Diagnostics: D0 real C0, D1 no-writeback checksum, D1b non-atomic stores,\n"
                      << "D2 atomic-only same-address-stream, D3 offline contention analysis.\n";
            return 0;
        }
        int groups = argc > 3 ? std::stoi(argv[3]) : 3;
        require(groups >= 3, "at least 3 measurement groups");
        check(cudaSetDevice(argc > 4 ? std::stoi(argv[4]) : 0));
        U swSize = argc > 6 ? U(std::stoi(argv[6])) : COLUMN_SW_SIZE;
        cudaStream_t stream; check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        auto m = readMatrix(argv[1]);
        const U features = U(argc > 2 ? std::stoi(argv[2]) : 128);
        auto col = convertColumn(m.rows, m.cols, m.csr, swSize);
        auto base = buildColumnBase(m.rows, m.cols, m.csr);
        auto x = readFeatures(argc > 5 ? argv[5] : nullptr, m.cols, features);
        Buffer<float> input(x), y(std::size_t(m.rows) * features);
        Buffer<std::uint32_t> win(col.colWindowOffset), slots(col.tileRowSlot), off(col.tileValueOffset),
            swOff(col.superWindowRowOffset), rows(col.superWindowRows), cacheSlots(col.superWindowRowCacheSlot),
            cachedRows(col.superWindowCachedRowSlot), counts(col.superWindowCachedCount);
        Buffer<std::uint64_t> bits(col.tileLocalBit);
        Buffer<float> vals(col.values);
        DeviceColumnSWTCF view{col.rows, col.cols, col.superWindowSize,
            win.data, slots.data, off.data, bits.data, vals.data,
            swOff.data, rows.data, cacheSlots.data, cachedRows.data, counts.data};
        // sink for D1 checksums: one float per (CW, feature-block) CTA.
        Buffer<float> sink(std::size_t((std::uint64_t(m.cols) + 7) / 8) *
            ((std::uint64_t(features) + COLUMN_FEATURE_TILE - 1) / COLUMN_FEATURE_TILE));

        // ---- D0: real C0 correctness (sampled) ----
        check(cudaMemsetAsync(y.data, 0, y.size * sizeof(float), stream));
        check(launchColumnSpMM(view, input.data, y.data, features, ColumnVersion::C0, stream, false));
        auto yHost = y.host(stream);
        const unsigned count = std::min(64u, m.rows);
        double maxAbs = 0;
        for (unsigned i = 0; i < count; ++i) {
            auto row = std::uint64_t(i) * (m.rows - 1) / std::max(1u, count - 1);
            for (U j = 0; j < features; ++j) {
                double sum = 0;
                for (auto e = m.csr.ptr[row]; e < m.csr.ptr[row + 1]; ++e)
                    sum += double(tf32(m.csr.values[e])) * tf32(x[std::size_t(m.csr.src[e]) * features + j]);
                maxAbs = std::max(maxAbs, std::fabs(double(yHost[row * features + j]) - sum));
            }
        }

        const int repeat = 5;
        const U windows = (std::uint64_t(m.cols) + 7) / 8;
        const U fblocks = (std::uint64_t(features) + COLUMN_FEATURE_TILE - 1) / COLUMN_FEATURE_TILE;
        dim3 grid(windows, fblocks);
        std::cout << std::setprecision(9);
        std::cout << "# column_output_diagnosis dataset=" << argv[1] << " M=" << m.rows << " K=" << m.cols
                  << " nnz=" << m.csr.src.size() << " F=" << features << " sw=" << swSize
                  << " groups=" << groups << " ctas=" << std::uint64_t(windows) * fblocks << '\n';
        std::cout << "diagnostic,kernel_ms,min,max,note\n";

        // D0 reference (correct kernel).
        {
            auto launch = [&]() { check(launchColumnSpMM(view, input.data, y.data, features, ColumnVersion::C0, stream, false)); };
            auto s = measureGraph(launch, repeat, groups, 3, stream);
            std::cout << "D0-C0," << s.med << ',' << s.mn << ',' << s.mx << ",sampled-max-abs-err=" << maxAbs << '\n' << std::flush;
        }
        // D1: no writeback.
        {
            auto launch = [&]() { diagColumnKernel<0><<<grid, 128, 0, stream>>>(view, input.data, y.data, sink.data, features); };
            auto s = measureGraph(launch, repeat, groups, 3, stream);
            std::cout << "D1-no-writeback," << s.med << ',' << s.mn << ',' << s.mx << ",checksum-keeps-address-chain\n" << std::flush;
        }
        // D1b: non-atomic scattered stores.
        {
            auto launch = [&]() { diagColumnKernel<1><<<grid, 128, 0, stream>>>(view, input.data, y.data, sink.data, features); };
            auto s = measureGraph(launch, repeat, groups, 3, stream);
            std::cout << "D1b-non-atomic-store," << s.med << ',' << s.mn << ',' << s.mx << ",diagnostic-only\n" << std::flush;
        }
        // D2: atomic-only, same address stream (REDG STRONG = what C0 emits).
        {
            auto launch = [&]() { diagColumnKernel<2><<<grid, 128, 0, stream>>>(view, nullptr, y.data, sink.data, features); };
            auto s = measureGraph(launch, repeat, groups, 3, stream);
            std::cout << "D2-atomic-strong," << s.med << ',' << s.mn << ',' << s.mx << ",REDG.STRONG-const-value\n" << std::flush;
        }
        // D2c: same op count on L2-resident contiguous scratch (control).
        {
            const auto d3atomic = [&]() {
                return analyzeD3(base, features).atomics;
            }();
            const std::uint64_t totalOps = d3atomic;
            const std::uint64_t threads = std::uint64_t(windows) * fblocks * 128;
            const U opsPerThread = U((totalOps + threads - 1) / threads);
            Buffer<float> scratch(0x400000); // 16 MiB
            auto launch = [&]() { diagAtomicContiguous<<<windows * fblocks, 128, 0, stream>>>(opsPerThread, scratch.data); };
            auto s = measureGraph(launch, repeat, groups, 3, stream);
            std::cout << "D2c-atomic-contiguous," << s.med << ',' << s.mn << ',' << s.mx
                      << ",L2-resident-scratch-logical-ops=" << totalOps << '\n' << std::flush;
        }
        // D3: offline analysis.
        auto d3 = analyzeD3(base, features);
        std::cout << std::setprecision(6);
        std::cout << "# D3 tiles=" << d3.tiles << " rows_with_updates=" << d3.rows
                  << " logical_atomics=" << d3.atomics << '\n';
        std::cout << "# D3 row-CW-contention p50/p95/p99/max=" << d3.p50 << '/' << d3.p95 << '/'
                  << d3.p99 << '/' << d3.maxCw << " mean=" << d3.meanCw << '\n';
        std::cout << "# D3 row-tile-updates mean/max=" << d3.meanTile << '/' << d3.maxTile << '\n';
        std::cout << "# D3 CW-index-spread-per-row mean/max=" << d3.meanSpread << '/' << d3.maxSpread << '\n';
        std::cout << "# D3 rows_with_>=2_CWs=" << d3.rowsContended2 << " rows_with_>=16_CWs=" << d3.rowsContended16
                  << " Y_working_set_MB=" << d3.yWorkingSetMB << '\n';
        check(cudaStreamDestroy(stream));
    } catch (const std::exception& e) { std::cerr << "Error: " << e.what() << '\n'; return 1; }
    return 0;
}
