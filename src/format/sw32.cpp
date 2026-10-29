#include "format/sw32.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>

namespace {
std::uint32_t checked32(std::size_t n) {
    if (n > std::numeric_limits<std::uint32_t>::max())
        throw std::overflow_error("source-mask format exceeds uint32 offsets");
    return static_cast<std::uint32_t>(n);
}
struct Cursor { std::uint32_t source, row, pos; };
struct Later {
    bool operator()(const Cursor& a, const Cursor& b) const {
        return a.source > b.source || (a.source == b.source && a.row > b.row);
    }
};
}

Sw32HostFormat convertCsrToSw32(std::uint32_t rows, std::uint32_t cols,
                                      const CsrMatrix& csr,
                                      bool bucketByRwMask) {
    if (!rows || !cols || csr.rowOffsets.size() != std::size_t(rows)+1 ||
        csr.rowOffsets.front() != 0 || csr.rowOffsets.back() != csr.sourceIds.size() ||
        csr.values.size() != csr.sourceIds.size())
        throw std::invalid_argument("invalid canonical CSR for source-mask SW32");
    for (std::uint32_t r = 0; r < rows; ++r)
        if (csr.rowOffsets[r] > csr.rowOffsets[r+1])
            throw std::invalid_argument("decreasing CSR offsets");
    Sw32HostFormat out;
    out.rows = rows;
    out.cols = cols;
    out.values.reserve(csr.values.size());
    out.sourceIds.reserve(csr.sourceIds.size());
    out.sourceMetadata.reserve(csr.sourceIds.size());
    out.windowOffsets.reserve((std::size_t(rows)+31)/32+1);
    std::priority_queue<Cursor, std::vector<Cursor>, Later> heap;
    struct Item { std::uint32_t source, mask, offset; };
    std::array<std::vector<Item>, 16> buckets;
    std::vector<float> windowValues;
    for (std::uint64_t base = 0; base < rows; base += 32) {
        for (auto& bucket : buckets) bucket.clear();
        windowValues.clear();
        const auto count = static_cast<std::uint32_t>(std::min<std::uint64_t>(32, rows-base));
        for (std::uint32_t r = 0; r < count; ++r) {
            const auto row = static_cast<std::uint32_t>(base)+r;
            const auto p = csr.rowOffsets[row];
            if (p != csr.rowOffsets[row+1]) heap.push({csr.sourceIds[p],r,p});
        }
        while (!heap.empty()) {
            const auto source = heap.top().source;
            if (source >= cols)
                throw std::invalid_argument("source-mask source outside column range");
            const auto offset = checked32(windowValues.size());
            std::uint32_t mask = 0;
            do {
                auto node = heap.top();
                heap.pop();
                const auto value = csr.values[node.pos];
                if (!std::isfinite(value) || value == 0)
                    throw std::invalid_argument("source-mask requires finite nonzero values");
                mask |= std::uint32_t(1) << node.row;
                windowValues.push_back(value);
                const auto row = static_cast<std::uint32_t>(base)+node.row;
                if (++node.pos < csr.rowOffsets[row+1]) {
                    node.source = csr.sourceIds[node.pos];
                    if (node.source <= source)
                        throw std::invalid_argument("source-mask requires increasing row sources");
                    heap.push(node);
                }
            } while (!heap.empty() && heap.top().source == source);
            const unsigned rowGroupMasks = unsigned((mask & 0xffu) != 0) |
                (unsigned((mask & 0xff00u) != 0) << 1) |
                (unsigned((mask & 0xff0000u) != 0) << 2) |
                (unsigned((mask & 0xff000000u) != 0) << 3);
            buckets[bucketByRwMask ? rowGroupMasks : 0].push_back({source, mask, offset});
        }
        for (unsigned bucketMask : sw32BucketOrder) {
            const auto& bucket = buckets[bucketMask];
            if (bucketByRwMask) out.bucketOffsets.push_back(checked32(out.rowGroupMasks.size()));
            unsigned slot = 0, rowGroupMasks = 0;
            for (const auto& item : bucket) {
                out.sourceIds.push_back(item.source);
                const auto valueBase = checked32(out.values.size());
                out.sourceMetadata.push_back(std::uint64_t(valueBase)<<32 | item.mask);
                const auto valueCount = __builtin_popcount(item.mask);
                out.values.insert(out.values.end(), windowValues.begin()+item.offset,
                                  windowValues.begin()+item.offset+valueCount);
                rowGroupMasks |= unsigned((item.mask & 0xffu) != 0) |
                    (unsigned((item.mask & 0xff00u) != 0) << 1) |
                    (unsigned((item.mask & 0xff0000u) != 0) << 2) |
                    (unsigned((item.mask & 0xff000000u) != 0) << 3);
                if (++slot == 8) {
                    out.rowGroupMasks.push_back(rowGroupMasks);
                    slot = rowGroupMasks = 0;
                }
            }
            if (slot) {
                while (slot++ < 8) {
                    out.sourceIds.push_back(UINT32_MAX);
                    out.sourceMetadata.push_back(0);
                }
                out.rowGroupMasks.push_back(rowGroupMasks);
            }
        }
        if (bucketByRwMask) out.bucketOffsets.push_back(checked32(out.rowGroupMasks.size()));
        out.windowOffsets.push_back(checked32(out.rowGroupMasks.size()));
    }
    checked32(out.values.size());
    if (bucketByRwMask) {
        const auto windows = checked32(out.windowOffsets.size()-1);
        out.tasks.reserve(windows);
        std::array<std::vector<std::uint32_t>, 4> rwParts;
        std::size_t parts = 0;
        for (std::uint32_t sw = 0; sw < windows; ++sw) {
            const auto begin = out.windowOffsets[sw], end = out.windowOffsets[sw+1];
            const auto count = end-begin;
            if (count <= sw32TaskPanelLimit) {
                out.tasks.push_back({sw, begin, end, UINT32_MAX, 15, 1, 0});
                continue;
            }
            for (auto& rw : rwParts) rw.clear();
            const auto chunks = 1+(count-1)/sw32TaskPanelLimit;
            const auto reduceFirst = checked32(out.reductions.size());
            for (auto panel = begin; panel < end;) {
                const auto panelEnd = panel+std::min(sw32TaskPanelLimit, end-panel);
                std::uint32_t mask = 0;
                for (unsigned slot = 1; slot < 16; ++slot) {
                    const auto first = std::size_t(sw)*17+slot;
                    if (std::max(panel, out.bucketOffsets[first]) <
                        std::min(panelEnd, out.bucketOffsets[first+1]))
                        mask |= sw32BucketOrder[slot];
                }
                out.tasks.push_back({sw, panel, panelEnd, checked32(parts), mask,
                                     chunks, reduceFirst});
                for (unsigned rw = 0; rw < 4; ++rw)
                    if (mask & (1u<<rw)) rwParts[rw].push_back(checked32(parts++));
                panel = panelEnd;
            }
            for (const auto& rw : rwParts) {
                const auto first = checked32(out.partialIds.size());
                out.partialIds.insert(out.partialIds.end(), rw.begin(), rw.end());
                out.reductions.push_back({first, checked32(out.partialIds.size())});
            }
        }
        checked32(out.tasks.size());
        checked32(out.reductions.size());
        out.partialCount = checked32(parts);
    }
    return out;
}
