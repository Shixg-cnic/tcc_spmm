// Adaptive (dynamic SuperWindow) correctness tests.
//
// Builds the SW-independent base format, applies several deterministic
// schedules (default grouping, aggressive small tasks, heavy-CW splitting),
// and checks:
//  - task table coverage: every CW/tile is owned by exactly one task;
//  - decoded adaptive format equals the CanonicalCSR (values + rows + cols);
//  - kernel output matches the CPU TF32 reference and the fixed-SW C3 kernel.
#include "column_schedule.cuh"
#include "column_test_utils.cuh"
#include <iomanip>

using namespace test;


namespace {


struct GPUAdaptive {
    Buffer<std::uint32_t> windows, slots, offset, swOffset, rows, cacheSlots, cachedRows, counts;
    Buffer<std::uint64_t> bits;
    Buffer<float> values;
    Buffer<SWTask> tasks;
    DeviceColumnAdaptive view;
    explicit GPUAdaptive(const ColumnAdaptiveFormat& m)
        : windows(m.colWindowOffset), slots(m.tileRowSlot), offset(m.tileValueOffset),
          swOffset(m.swRowOffset), rows(m.swRows), cacheSlots(m.swRowCacheSlot),
          cachedRows(m.swCachedRowSlot), counts(m.swCachedCount), bits(m.tileLocalBit), values(m.values),
          tasks(m.tasks),
          view{m.rows, m.cols, uint32_t(m.tasks.size()), windows.data, slots.data, offset.data, bits.data,
               values.data, swOffset.data, rows.data, cacheSlots.data, cachedRows.data, counts.data, tasks.data} {}
};

void verifyCoverage(const ColumnBaseFormat& base, const ColumnSchedule& sched) {
    const std::uint32_t W = base.colWindowOffset.size()-1;
    std::vector<std::uint64_t> tileOwners(base.tileLocalBit.size(), 0);
    std::uint32_t prevCwBegin = 0, prevCwEnd = 0;
    for (auto& t : sched.tasks) {
        require(t.cwBegin < t.cwEnd && t.cwEnd <= W, "task CW range out of bounds");
        require(t.cwBegin >= prevCwBegin, "task CW ranges overlap or gap");
        prevCwBegin = t.cwBegin;
        prevCwEnd = std::max(prevCwEnd, t.cwEnd);
        require(t.tileBegin <= t.tileEnd, "task tile range negative");
        for (std::uint32_t cw = t.cwBegin; cw < t.cwEnd; ++cw) {
            auto lo = std::max<std::uint64_t>(t.tileBegin, base.colWindowOffset[cw]);
            auto hi = std::min<std::uint64_t>(t.tileEnd, base.colWindowOffset[cw+1]);
            for (auto tile = lo; tile < hi; ++tile) ++tileOwners[tile];
        }
    }
    require(prevCwEnd == W, "tasks do not cover all CWs");
    for (std::size_t i = 0; i < tileOwners.size(); ++i)
        require(tileOwners[i] == 1, "tile not owned exactly once");
}

void decodeAdaptive(const ColumnAdaptiveFormat& m, const CanonicalCSR& csr) {
    std::vector<std::vector<std::pair<unsigned, float>>> entries(m.rows);
    for (auto& t : m.tasks) {
        const auto* swRows = m.swRows.data() + t.rowBase;
        for (std::uint32_t cw = t.cwBegin; cw < t.cwEnd; ++cw) {
            auto lo = std::max<std::uint64_t>(t.tileBegin, m.colWindowOffset[cw]);
            auto hi = std::min<std::uint64_t>(t.tileEnd, m.colWindowOffset[cw+1]);
            for (auto tile = lo; tile < hi; ++tile) {
                auto v = m.tileValueOffset[tile];
                auto bits = m.tileLocalBit[tile];
                for (unsigned bit = 0; bit < 64; ++bit) if (bits & (1ULL<<bit)) {
                    auto slot = m.tileRowSlot[std::size_t(tile)*8 + bit/8];
                    require(slot != UINT32_MAX, "nonzero in padded slot");
                    auto row = swRows[slot];
                    entries.at(row).emplace_back(cw*8 + bit%8, m.values.at(v++));
                }
                require(v == m.tileValueOffset[tile+1], "value/bitmap ordering mismatch");
            }
        }
    }
    for (unsigned r = 0; r < m.rows; ++r) {
        std::sort(entries[r].begin(), entries[r].end());
        require(entries[r].size() == csr.ptr[r+1]-csr.ptr[r], "decoded row length mismatch");
        for (std::size_t i = 0; i < entries[r].size(); ++i) {
            auto e = csr.ptr[r]+i;
            require(entries[r][i].first == csr.src[e] && entries[r][i].second == csr.values[e], "decoded mismatch");
        }
    }
}

void runMatrix(const Matrix& matrix, unsigned f, cudaStream_t stream,
               const std::vector<ScheduleConfig>& configs, bool checkVsC3) {
    auto base = buildColumnBase(matrix.rows, matrix.cols, matrix.csr);
    auto x = features(matrix.cols, f);
    auto expected = reference(matrix, x, f, true);
    Buffer<float> input(x);
    for (auto& cfg : configs) {
        auto sched = buildAdaptiveSchedule(base, cfg);
        verifyCoverage(base, sched);
        auto fmt = buildAdaptiveColumn(base, sched);
        decodeAdaptive(fmt, matrix.csr);
        GPUAdaptive ga(fmt);
        Buffer<float> y(std::size_t(matrix.rows) * f);
        check(launchColumnAdaptiveSpMM(ga.view, input.data, y.data, f, true, stream));
        auto out = y.host(stream);
        Buffer<float> yn(std::size_t(matrix.rows) * f);
        check(launchColumnAdaptiveSpMM(ga.view, input.data, yn.data, f, false, stream));
        auto outN = yn.host(stream);
        auto eRef = error(out, expected);
        if (checkVsC3) {
            auto col = convertColumn(matrix.rows, matrix.cols, matrix.csr);
            GPUColumn gc(col);
            Buffer<float> y3(std::size_t(matrix.rows) * f);
            check(launchColumnSpMM(gc.view, input.data, y3.data, f, ColumnVersion::C3, stream));
            auto out3 = y3.host(stream);
            auto e = error(out, out3);
            require(e.rmse < 2e-4 && e.maxAbsolute < 2e-4 + e.maxReference * 5e-4, "adaptive vs C3 mismatch");
        }
        verify(eRef);
        verify(error(outN, expected));
        std::cout << "M=" << matrix.rows << " K=" << matrix.cols << " F=" << f
                  << " tasks=" << fmt.tasks.size() << " split_chunks=" << sched.splitChunks
                  << " config={maxTiles=" << cfg.maxTilesPerTask << ",splitThresh=" << cfg.splitTileThreshold
                  << ",chunk=" << cfg.splitChunkTiles << ",maxCws=" << cfg.maxCwsPerSw
                  << ",reuseFrac=" << cfg.minReuseSavingFraction << "} OK\n";
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        cudaStream_t stream; check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        ScheduleConfig splitAggressive;
        splitAggressive.splitTileThreshold = 4; splitAggressive.splitChunkTiles = 2;
        splitAggressive.maxTilesPerTask = 8; splitAggressive.maxCwsPerSw = 2;
        splitAggressive.minReuseSavingFraction = 0.0;
        ScheduleConfig tinyTasks;
        tinyTasks.splitTileThreshold = UINT32_MAX; tinyTasks.maxTilesPerTask = 24;
        tinyTasks.maxCwsPerSw = 2; tinyTasks.minReuseSavingFraction = 0.0;
        ScheduleConfig noMerge;
        noMerge.splitTileThreshold = UINT32_MAX; noMerge.maxCwsPerSw = 1;
        std::vector<ScheduleConfig> configs{ ScheduleConfig{}, splitAggressive, tinyTasks, noMerge };

        auto weighted = makeMatrix(19, 269, [](unsigned r, unsigned c) {
            return ((r<8 && c<8) || ((r*31+c*7)%13<3)) ? float(int((r*17+c*3)%41)-20)/19.f : 0.f;
        });
        for (unsigned f : {1u, 17u, 32u, 64u, 100u, 128u, 256u, 512u, 513u, 544u, 1024u})
            runMatrix(weighted, f, stream, configs, true);
        // a matrix with dense column windows to exercise heavy-CW splits
        auto dense = makeMatrix(65, 401, [](unsigned r, unsigned c) { return c<8 || c>=384 ? 1.f : 0.f; });
        runMatrix(dense, 64, stream, configs, true);
        // zero matrix / empty CWs
        runMatrix(makeMatrix(9, 259, [](unsigned, unsigned) { return 0.f; }), 100, stream, configs, false);
        // repeated rows (cache pressure)
        auto repeated = makeMatrix(65, 265, [](unsigned, unsigned c) { return c%8==0 ? 1.f : 0.f; });
        runMatrix(repeated, 100, stream, configs, true);
        // rectangle, negative weights, odd F
        runMatrix(makeMatrix(17, 273, [](unsigned r, unsigned c) { return r==16 && c==272 ? -1.25f : 0.f; }),
                  17, stream, configs, false);
        std::cout << "adaptive_sw_test: all configurations passed\n";
        check(cudaStreamDestroy(stream));
    } catch (const std::exception& e) { std::cerr << "Error: " << e.what() << '\n'; return 1; }
    return 0;
}
