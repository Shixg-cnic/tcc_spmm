// Offline structural analysis of the Column-SWTCF format.
//
// Computes, from the REAL format data (never estimated from average degree):
//  - atomic update counts before caching, after the fixed 32-row cache, and
//    the theoretical maximum saving with an unlimited cache;
//  - per-CW and per-SW (per CTA) tile workload distributions;
//  - output-row repetition statistics inside each SW.
//
// One pass per SW size in {1,2,4,8,16,32,64}; output CSVs are written to
// results/baseline/. CPU-only analysis; kernels are not launched.
#include "matrix_io.cuh"
#include <iomanip>
#include <fstream>
#include <numeric>

using namespace test;

namespace {

struct SwStats {
    std::uint64_t occurrences = 0;          // total (CW,row) pairs
    std::uint64_t uniqueRows = 0, repeatRows = 0;
    std::uint64_t atomicBefore = 0, atomicAfterCache32 = 0, atomicMaxSaved = 0;
    std::uint64_t swCount = 0, ctaTilesMax = 0;
    std::vector<std::uint64_t> ctaTiles, cwTiles, cwRows;
    std::uint64_t emptyCw = 0;
};

SwStats analyze(const ColumnSWTCF& m) {
    SwStats s;
    s.swCount = m.superWindowCachedCount.size();
    for (std::size_t w = 0; w + 1 < m.colWindowOffset.size(); ++w) {
        auto tiles = m.colWindowOffset[w + 1] - m.colWindowOffset[w];
        s.cwTiles.push_back(tiles);
        std::uint64_t rows = 0;
        for (auto t = m.colWindowOffset[w]; t < m.colWindowOffset[w + 1]; ++t)
            for (unsigned r = 0; r < 8; ++r)
                rows += m.tileRowSlot[std::size_t(t) * 8 + r] != UINT32_MAX;
        s.cwRows.push_back(rows);
        if (!tiles) ++s.emptyCw;
    }
    std::uint64_t ctaTilesSum = 0;
    for (std::size_t sw = 0; sw < s.swCount; ++sw) {
        const auto first = std::uint64_t(sw) * m.superWindowSize;
        const auto end = std::min<std::uint64_t>(first + m.superWindowSize, m.colWindowOffset.size() - 1);
        std::uint64_t swTiles = 0;
        for (auto w = first; w < end; ++w) swTiles += m.colWindowOffset[w + 1] - m.colWindowOffset[w];
        s.ctaTiles.push_back(swTiles);
        s.ctaTilesMax = std::max(s.ctaTilesMax, swTiles);
        ctaTilesSum += swTiles;

        auto rowBase = m.superWindowRowOffset[sw];
        auto count = m.superWindowRowOffset[sw + 1] - rowBase;
        std::vector<unsigned> freq(count, 0);
        for (auto w = first; w < end; ++w)
            for (auto t = m.colWindowOffset[w]; t < m.colWindowOffset[w + 1]; ++t)
                for (unsigned r = 0; r < 8; ++r) {
                    auto slot = m.tileRowSlot[std::size_t(t) * 8 + r];
                    if (slot != UINT32_MAX) { ++freq[slot]; ++s.occurrences; }
                }
        std::uint64_t swUnique = 0, swRepeat = 0, swSaved32 = 0, swMaxSaved = 0;
        for (unsigned r = 0; r < count; ++r) {
            ++swUnique;
            if (freq[r] > 1) { ++swRepeat; swMaxSaved += freq[r] - 1; }
            if (m.superWindowRowCacheSlot[rowBase + r] != UINT32_MAX) swSaved32 += freq[r] - 1;
        }
        s.uniqueRows += swUnique;
        s.repeatRows += swRepeat;
        s.atomicMaxSaved += swMaxSaved;
        s.atomicAfterCache32 += swSaved32;
    }
    s.atomicBefore = s.occurrences;
    s.atomicAfterCache32 = s.occurrences - s.atomicAfterCache32; // F multiplies outside
    return s;
}

double percentile(std::vector<std::uint64_t> v, double p) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[std::size_t(std::min<double>(v.size() - 1, std::max<double>(0, p * (v.size() - 1))))];
}

void writeSw(const std::string& path, const std::vector<std::pair<unsigned, SwStats>>& rows, unsigned f) {
    std::ofstream out(path);
    out << "sw_size,total_sw,total_tiles,total_row_occurrences,unique_rows_per_sw,repeat_rows_per_sw,"
           "atomic_before,atomic_after_cache32,atomic_saved_ratio,max_possible_atomic_saved_ratio,"
           "cta_tiles_mean,cta_tiles_p50,cta_tiles_p95,cta_tiles_p99,cta_tiles_max,top1pct_cta_tiles_share,"
           "empty_cta_count,cw_tiles_mean,cw_tiles_p50,cw_tiles_p95,cw_tiles_p99,cw_tiles_max,"
           "top1pct_cw_tiles_share,empty_cw_count,cw_rows_mean,cw_rows_p50,cw_rows_p95,cw_rows_p99,cw_rows_max\n";
    for (auto& [sw, s] : rows) {
        std::uint64_t totalTiles = 0;
        for (auto t : s.cwTiles) totalTiles += t;
        double top1Cta = 0, ctaTotal = 0;
        auto ctaSorted = s.ctaTiles; std::sort(ctaSorted.begin(), ctaSorted.end());
        auto cwSorted = s.cwTiles; std::sort(cwSorted.begin(), cwSorted.end());
        auto cwRowsSorted = s.cwRows; std::sort(cwRowsSorted.begin(), cwRowsSorted.end());
        for (auto t : ctaSorted) ctaTotal += t;
        for (std::size_t i = ctaSorted.size() - ctaSorted.size() / 100; i < ctaSorted.size(); ++i) top1Cta += ctaSorted[i];
        double top1Cw = 0, cwTotal = 0;
        for (auto t : cwSorted) cwTotal += t;
        for (std::size_t i = cwSorted.size() - cwSorted.size() / 100; i < cwSorted.size(); ++i) top1Cw += cwSorted[i];
        auto mean = [](const std::vector<std::uint64_t>& v) { return v.empty() ? 0.0 : double(std::accumulate(v.begin(), v.end(), 0ULL)) / v.size(); };
        out << sw << ',' << s.swCount << ',' << totalTiles << ',' << s.occurrences << ','
            << double(s.uniqueRows) / s.swCount << ',' << double(s.repeatRows) / s.swCount << ','
            << s.atomicBefore * f << ',' << s.atomicAfterCache32 * f << ','
            << (s.atomicBefore ? double(s.atomicBefore - s.atomicAfterCache32) / s.atomicBefore : 0) << ','
            << (s.atomicBefore ? double(s.atomicMaxSaved) / s.atomicBefore : 0) << ','
            << mean(s.ctaTiles) << ',' << percentile(s.ctaTiles, 0.50) << ',' << percentile(s.ctaTiles, 0.95) << ','
            << percentile(s.ctaTiles, 0.99) << ',' << s.ctaTilesMax << ','
            << (ctaTotal ? top1Cta / ctaTotal : 0) << ','
            << std::count(s.ctaTiles.begin(), s.ctaTiles.end(), 0ULL) << ','
            << mean(s.cwTiles) << ',' << percentile(s.cwTiles, 0.50) << ',' << percentile(s.cwTiles, 0.95) << ','
            << percentile(s.cwTiles, 0.99) << ',' << (cwSorted.empty() ? 0 : cwSorted.back()) << ','
            << (cwTotal ? top1Cw / cwTotal : 0) << ',' << s.emptyCw << ','
            << mean(s.cwRows) << ',' << percentile(s.cwRows, 0.50) << ',' << percentile(s.cwRows, 0.95) << ','
            << percentile(s.cwRows, 0.99) << ',' << (cwRowsSorted.empty() ? 0 : cwRowsSorted.back()) << '\n';
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            std::cerr << "Usage: analyze_structure matrix.csr [F=128] [out_prefix=results/baseline/struct]\n";
            return 1;
        }
        auto m = readMatrix(argv[1]);
        unsigned f = argc > 2 ? unsigned(std::stoi(argv[2])) : 128;
        std::string prefix = argc > 3 ? argv[3] : std::string("results/baseline/struct");
        std::cout << "# analyze_structure " << argv[1] << " M=" << m.rows << " K=" << m.cols
                  << " nnz=" << m.csr.src.size() << " F=" << f << '\n';
        std::vector<std::pair<unsigned, SwStats>> rows;
        for (unsigned sw : {1u, 2u, 4u, 8u, 16u, 32u, 64u}) {
            auto t0 = Clock::now();
            auto col = convertColumn(m.rows, m.cols, m.csr, sw);
            rows.emplace_back(sw, analyze(col));
            std::cout << "SW=" << sw << " convert_ms=" << elapsed(t0)
                      << " total_tiles=" << rows.back().second.cwTiles.size() << '\n';
        }
        writeSw(prefix + "_atomic_workload.csv", rows, f);
        std::cout << "wrote " << prefix << "_atomic_workload.csv\n";
    } catch (const std::exception& e) { std::cerr << "Error: " << e.what() << '\n'; return 1; }
    return 0;
}
