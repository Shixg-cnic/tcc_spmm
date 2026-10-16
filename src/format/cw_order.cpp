// Offline CW execution order builders + offline locality statistics.
// See include/cw_order.cuh. CPU-only preprocessing; no GPU work here.
//
// Complexity (products ~306k CWs / ~2.45M rows):
//   rowFreq          : O(total effective (CW,row) pairs)
//   anchors/bands    : same, plus O(W) histograms of width bandCount
//   sort             : O(W log W)
//   group reuse      : O(total effective (CW,row) pairs) per permutation
// No WxW similarity matrix, no per-CW dense row bitmaps, no graph partitioner.
#include "cw_order.cuh"

#include <algorithm>
#include <chrono>
#include <numeric>
#include <random>
#include <stdexcept>

namespace {

using Clock = std::chrono::steady_clock;
inline double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// Effective (CW,row) pairs: every real row of every tile of every CW, exactly
// once per CW (buildColumnBase guarantees a CW contains each real row once).
inline std::uint64_t countRowOccurrences(const ColumnBaseFormat& m) {
    std::uint64_t n = 0;
    for (std::uint32_t w = 0; w + 1 < m.colWindowOffset.size(); ++w)
        for (auto t = m.colWindowOffset[w]; t < m.colWindowOffset[w + 1]; ++t)
            for (unsigned r = 0; r < 8; ++r)
                if (m.tileRows[std::size_t(t) * 8 + r] != UINT32_MAX) ++n;
    return n;
}

// Per-CW output row set as a sorted vector (empty CW -> empty).
inline std::vector<std::uint32_t> cwRows(const ColumnBaseFormat& m, std::uint32_t w) {
    std::vector<std::uint32_t> rows;
    for (auto t = m.colWindowOffset[w]; t < m.colWindowOffset[w + 1]; ++t)
        for (unsigned r = 0; r < 8; ++r) {
            auto row = m.tileRows[std::size_t(t) * 8 + r];
            if (row != UINT32_MAX) rows.push_back(row);
        }
    std::sort(rows.begin(), rows.end());
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    return rows;
}

std::vector<std::uint32_t> buildRowFreq(const ColumnBaseFormat& m) {
    std::vector<std::uint32_t> freq(m.rows, 0);
    for (std::uint32_t w = 0; w + 1 < m.colWindowOffset.size(); ++w) {
        for (auto t = m.colWindowOffset[w]; t < m.colWindowOffset[w + 1]; ++t) {
            for (unsigned r = 0; r < 8; ++r) {
                auto row = m.tileRows[std::size_t(t) * 8 + r];
                if (row != UINT32_MAX) ++freq[row];
            }
        }
    }
    return freq;
}
} // namespace

CWOrderResult buildCWOrder(const ColumnBaseFormat& matrix, const CWOrderConfig& config) {
    const std::uint32_t W = matrix.colWindowOffset.size() - 1;
    CWOrderResult res;
    res.order.resize(W);
    auto t0 = Clock::now();
    const std::uint32_t bandRows = config.bandRows ? config.bandRows : 1;
    const std::uint32_t bandCount = (matrix.rows + bandRows - 1) / bandRows;

    switch (config.strategy) {
        case CWOrderStrategy::Identity: {
            std::iota(res.order.begin(), res.order.end(), 0);
            break;
        }
        case CWOrderStrategy::SeededRandom: {
            std::iota(res.order.begin(), res.order.end(), 0);
            std::mt19937 rng(config.seed);
            std::shuffle(res.order.begin(), res.order.end(), rng);
            break;
        }
        case CWOrderStrategy::HotAnchor: {
            auto freq = buildRowFreq(matrix);
            std::vector<std::uint32_t> anchor(W, UINT32_MAX);
            std::iota(res.order.begin(), res.order.end(), 0);
            for (std::uint32_t w = 0; w < W; ++w) {
                std::uint32_t bestRow = UINT32_MAX, bestFreq = 0;
                for (auto t = matrix.colWindowOffset[w]; t < matrix.colWindowOffset[w + 1]; ++t)
                    for (unsigned r = 0; r < 8; ++r) {
                        auto row = matrix.tileRows[std::size_t(t) * 8 + r];
                        if (row == UINT32_MAX) continue;
                        auto f = freq[row];
                        if (f > bestFreq || (f == bestFreq && row < bestRow)) { bestFreq = f; bestRow = row; }
                    }
                anchor[w] = bestRow; // UINT32_MAX for empty CW -> sorts last
            }
            std::sort(res.order.begin(), res.order.end(), [&](std::uint32_t a, std::uint32_t b) {
                if (anchor[a] != anchor[b]) return anchor[a] < anchor[b];
                return a < b;
            });
            break;
        }
        case CWOrderStrategy::BandAnchor: {
            // Pass 1: row frequency (needed for the anchor tie-break key).
            auto freq = buildRowFreq(matrix);
            std::vector<std::uint32_t> dominant(W, UINT32_MAX), anchor(W, UINT32_MAX);
            std::iota(res.order.begin(), res.order.end(), 0);
            std::vector<std::uint32_t> hist(bandCount, 0);
            for (std::uint32_t w = 0; w < W; ++w) {
                std::fill(hist.begin(), hist.end(), 0);
                std::uint32_t bestRow = UINT32_MAX, bestFreq = 0;
                std::uint32_t bestBand = UINT32_MAX, bestBandCount = 0;
                for (auto t = matrix.colWindowOffset[w]; t < matrix.colWindowOffset[w + 1]; ++t)
                    for (unsigned r = 0; r < 8; ++r) {
                        auto row = matrix.tileRows[std::size_t(t) * 8 + r];
                        if (row == UINT32_MAX) continue;
                        auto f = freq[row];
                        if (f > bestFreq || (f == bestFreq && row < bestRow)) { bestFreq = f; bestRow = row; }
                        auto b = row / bandRows;
                        if (++hist[b] > bestBandCount) { bestBandCount = hist[b]; bestBand = b; }
                        // equal count keeps the smaller band: first hit wins
                    }
                anchor[w] = bestRow;
                dominant[w] = bestBand; // UINT32_MAX for empty CW -> sorts last
            }
            std::sort(res.order.begin(), res.order.end(), [&](std::uint32_t a, std::uint32_t b) {
                if (dominant[a] != dominant[b]) return dominant[a] < dominant[b];
                if (anchor[a] != anchor[b]) return anchor[a] < anchor[b];
                return a < b;
            });
            break;
        }
        default:
            throw std::invalid_argument("unknown CWOrderStrategy");
    }
    res.buildMs = elapsed(t0);
    return res;
}

void validateCWOrder(const std::vector<std::uint32_t>& order, std::uint32_t numWindows) {
    if (order.size() != numWindows)
        throw std::runtime_error("taskOrder size != numWindows");
    std::vector<std::uint8_t> seen(numWindows, 0);
    for (auto w : order) {
        if (w >= numWindows) throw std::runtime_error("CW id out of range");
        if (seen[w]) throw std::runtime_error("duplicated CW");
        seen[w] = 1;
    }
    for (std::uint32_t w = 0; w < numWindows; ++w)
        if (!seen[w]) throw std::runtime_error("missing CW in taskOrder");
}

// p95 of a sorted vector of group sizes (quantile helper).
static double pct95(const std::vector<double>& v) {
    if (v.empty()) return 0;
    return v[std::size_t(0.95 * (v.size() - 1))];
}

CWOrderStructureStats analyzeCWOrder(const ColumnBaseFormat& matrix,
    const std::vector<std::uint32_t>& order, std::uint32_t bandRows,
    const std::uint32_t groupSizes[3]) {
    CWOrderStructureStats st;
    st.numWindows = matrix.colWindowOffset.size() - 1;
    st.totalTiles = matrix.colWindowOffset.back();
    st.totalRowOccurrences = countRowOccurrences(matrix);

    // dominant-band concentration (BandAnchor metric, independent of order).
    if (st.numWindows) {
        const std::uint32_t bandRowsC = bandRows ? bandRows : 1;
        const std::uint32_t bandCount = (matrix.rows + bandRowsC - 1) / bandRowsC;
        std::vector<std::uint32_t> hist(bandCount, 0);
        double sum = 0;
        std::uint64_t valid = 0;
        for (std::uint32_t w = 0; w < st.numWindows; ++w) {
            std::fill(hist.begin(), hist.end(), 0);
            std::uint32_t rowsN = 0, maxB = 0;
            for (auto t = matrix.colWindowOffset[w]; t < matrix.colWindowOffset[w + 1]; ++t)
                for (unsigned r = 0; r < 8; ++r) {
                    auto row = matrix.tileRows[std::size_t(t) * 8 + r];
                    if (row == UINT32_MAX) continue;
                    ++rowsN;
                    auto b = row / bandRowsC;
                    auto c = ++hist[b];
                    if (c > maxB) maxB = c;
                }
            if (rowsN) { sum += double(maxB) / double(rowsN); ++valid; }
        }
        st.dominantBandConcentration = valid ? sum / double(valid) : 0;
    }

    const std::uint32_t groupSizesArr[3] = {32, 64, 128};
    for (int gi = 0; gi < 3; ++gi) {
        const std::uint32_t G = groupSizes ? groupSizes[gi] : groupSizesArr[gi];
        CWReuseStats& rs = st.reuse[gi];
        rs.groupSize = G;
        std::vector<uint32_t> lastSeen(matrix.rows, UINT32_MAX);
        std::vector<double> uniqPerGroup, yBytesPerGroup;
        std::uint64_t visits = 0, unique = 0;
        std::uint32_t groupId = 0;
        const std::uint32_t W = st.numWindows;
        for (std::uint32_t i = 0; i < W; ++i) {
            if (i % G == 0) { if (i) { uniqPerGroup.push_back(double(unique)); } ++groupId; unique = 0; }
            const std::uint32_t w = order[i];
            for (auto t = matrix.colWindowOffset[w]; t < matrix.colWindowOffset[w + 1]; ++t)
                for (unsigned r = 0; r < 8; ++r) {
                    auto row = matrix.tileRows[std::size_t(t) * 8 + r];
                    if (row == UINT32_MAX) continue;
                    ++visits;
                    if (lastSeen[row] != groupId) { lastSeen[row] = groupId; ++unique; }
                }
        }
        if (W % G == 0 || W == 0) { /* trailing group may be partial; record it */ }
        if (W) uniqPerGroup.push_back(double(unique));
        std::sort(uniqPerGroup.begin(), uniqPerGroup.end());
        rs.uniqueRowsMean = uniqPerGroup.empty() ? 0 : std::accumulate(uniqPerGroup.begin(), uniqPerGroup.end(), 0.0) / uniqPerGroup.size();
        rs.uniqueRowsP95 = pct95(uniqPerGroup);
        rs.uniqueRowsMax = uniqPerGroup.empty() ? 0 : uniqPerGroup.back();
        // Y working bytes reported per F (caller emits one row per F).
        auto bytesOf = [](double u, double F) { return u * F * 4.0; };
        rs.yBytesMean = bytesOf(rs.uniqueRowsMean, 128);
        rs.yBytesP95 = bytesOf(rs.uniqueRowsP95, 128);
        rs.yBytesMax = bytesOf(rs.uniqueRowsMax, 128);
        rs.reuseRatio = visits ? double(visits - std::uint64_t(std::accumulate(uniqPerGroup.begin(), uniqPerGroup.end(), 0.0))) / double(visits) : 0;
        (void)yBytesPerGroup;
    }
    return st;
}
