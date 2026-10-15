#include "column_schedule.cuh"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

namespace {
std::uint32_t checked(std::size_t n) {
    if (n >= UINT32_MAX) throw std::overflow_error("adaptive format 32-bit index overflow");
    return static_cast<std::uint32_t>(n);
}

// Sorted unique real rows covered by absolute tiles [tileBegin, tileEnd).
std::vector<std::uint32_t> tileRangeRows(const ColumnBaseFormat& base,
                                         std::uint32_t tileBegin, std::uint32_t tileEnd) {
    std::vector<std::uint32_t> rows;
    for (auto t = tileBegin; t < tileEnd; ++t)
        for (unsigned r = 0; r < 8; ++r) {
            auto row = base.tileRows[std::size_t(t) * 8 + r];
            if (row != UINT32_MAX) rows.push_back(row);
        }
    std::sort(rows.begin(), rows.end());
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    return rows;
}
} // namespace

ColumnBaseFormat buildColumnBase(std::uint32_t rows, std::uint32_t cols, const CanonicalCSR& csr) {
    if (!rows || !cols || csr.ptr.size() != std::size_t(rows)+1 ||
        csr.ptr.front() || csr.ptr.back() != csr.src.size() || csr.values.size() != csr.src.size())
        throw std::invalid_argument("invalid base CSR dimensions/offsets");
    checked(csr.src.size());
    const std::uint32_t windows = (std::uint64_t(cols)+7)/8;
    std::vector<std::size_t> offsets(std::size_t(windows)+1,0);
    for (std::uint32_t row=0; row<rows; ++row) {
        if (csr.ptr[row] > csr.ptr[row+1] || csr.ptr[row+1] > csr.src.size())
            throw std::invalid_argument("invalid CSR offset");
        for (auto e=csr.ptr[row]; e<csr.ptr[row+1]; ++e) {
            if (csr.src[e] >= cols || !std::isfinite(csr.values[e]) || csr.values[e] == 0 ||
                (e>csr.ptr[row] && csr.src[e]<=csr.src[e-1]))
                throw std::invalid_argument("buildColumnBase requires canonical CSR");
            ++offsets[csr.src[e]/8+1];
        }
    }
    for (std::uint32_t w=0; w<windows; ++w) offsets[w+1]+=offsets[w];
    auto cursor=offsets;
    struct Edge { std::uint32_t row, col; float value; };
    std::vector<Edge> edges(csr.src.size());
    for (std::uint32_t row=0; row<rows; ++row)
        for (auto e=csr.ptr[row]; e<csr.ptr[row+1]; ++e)
            edges[cursor[csr.src[e]/8]++] = {row,csr.src[e]%8,csr.values[e]};
    ColumnBaseFormat out;
    out.rows=rows; out.cols=cols;
    for (std::uint32_t w=0; w<windows; ++w) {
        std::uint64_t bitmap=0;
        std::uint32_t localRow=0;
        for (auto e=offsets[w]; e<offsets[w+1]; ++e) {
            if (e==offsets[w] || edges[e].row!=edges[e-1].row) {
                if (localRow==8) {
                    out.tileLocalBit.push_back(bitmap);
                    out.tileValueOffset.push_back(checked(out.values.size()));
                    bitmap=0; localRow=0;
                }
                checked(out.tileRows.size()+1);
                out.tileRows.push_back(edges[e].row); ++localRow;
            }
            bitmap |= std::uint64_t{1} << ((localRow-1)*8+edges[e].col);
            out.values.push_back(edges[e].value);
        }
        if (localRow) {
            checked(out.tileRows.size()+8-localRow);
            while (localRow++<8) out.tileRows.push_back(UINT32_MAX);
            out.tileLocalBit.push_back(bitmap);
            out.tileValueOffset.push_back(checked(out.values.size()));
        }
        out.colWindowOffset.push_back(checked(out.tileLocalBit.size()));
    }
    return out;
}

ColumnSchedule buildAdaptiveSchedule(const ColumnBaseFormat& base, const ScheduleConfig& config) {
    const std::uint32_t W = base.colWindowOffset.size()-1;
    std::vector<std::vector<std::uint32_t>> cwRows(W);
    std::vector<std::uint32_t> cwTiles(W), cwFirst(W), cwLast(W);
    for (std::uint32_t w=0; w<W; ++w) {
        cwFirst[w]=base.colWindowOffset[w];
        cwLast[w]=base.colWindowOffset[w+1];
        cwTiles[w]=cwLast[w]-cwFirst[w];
        cwRows[w]=tileRangeRows(base, cwFirst[w], cwLast[w]);
    }
    auto groupRows=[&](std::uint32_t a, std::uint32_t b) {
        std::vector<std::uint32_t> rows;
        for (auto w=a; w<=b; ++w) rows.insert(rows.end(), cwRows[w].begin(), cwRows[w].end());
        std::sort(rows.begin(), rows.end());
        rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
        return rows;
    };
    // Build one SW's row mapping + cache directory for the given CW range/tiles.
    auto makeSw=[&](std::uint32_t cwBegin, std::uint32_t cwEnd, std::uint32_t swId,
                    std::uint32_t tileBegin, std::uint32_t tileEnd, std::uint32_t flags,
                    ColumnSchedule& sched, std::vector<std::uint32_t>* swRowsOut) {
        SWTask task;
        task.cwBegin=cwBegin; task.cwEnd=cwEnd;
        task.tileBegin=tileBegin; task.tileEnd=tileEnd;
        task.swId=swId; task.flags=flags;
        std::vector<std::uint32_t> rows;
        if (flags==SWT_SPLIT_CHUNK) rows=tileRangeRows(base, tileBegin, tileEnd);
        else rows=groupRows(cwBegin, cwEnd-1);
        task.rowBase=checked(sched.swRows.size());
        sched.swRows.insert(sched.swRows.end(), rows.begin(), rows.end());
        sched.swRowOffset.push_back(checked(sched.swRows.size()));
        if (swRowsOut) *swRowsOut=rows;
        // cache selection: rows appearing in >1 CW of the group (split chunks: none)
        std::vector<std::uint32_t> slots(rows.size(), UINT32_MAX);
        std::uint32_t count=0;
        if (flags!=SWT_SPLIT_CHUNK) {
            std::unordered_map<std::uint32_t,unsigned> freqMap;
            for (auto w=cwBegin; w<cwEnd; ++w)
                for (auto r : cwRows[w]) ++freqMap[r];
            std::vector<std::pair<unsigned,std::uint32_t>> cand; // (freq,row)
            for (auto r : rows) { auto f=freqMap[r]; if (f>1) cand.emplace_back(f,r); }
            std::sort(cand.begin(), cand.end(), [](auto a, auto b) {
                return a.first!=b.first ? a.first>b.first : a.second<b.second;
            });
            for (auto& [f, r] : cand) {
                if (count>=COLUMN_CACHE_SIZE) break;
                auto it=std::lower_bound(rows.begin(), rows.end(), r);
                slots[std::size_t(it-rows.begin())]=count;
                sched.swCachedRowSlot.push_back(std::uint32_t(it-rows.begin()));
                ++count;
            }
        }
        // swRowOffset is seeded with {0}, so the per-SW padding target is
        // (swRowOffset.size()-1)*32 = (swId+1)*32 sections.
        while (sched.swCachedRowSlot.size() < (sched.swRowOffset.size()-1)*COLUMN_CACHE_SIZE)
            sched.swCachedRowSlot.push_back(UINT32_MAX);
        sched.swCachedCount.push_back(count);
        sched.swRowCacheSlot.insert(sched.swRowCacheSlot.end(), slots.begin(), slots.end());
        return task;
    };

    ColumnSchedule sched;
    std::uint32_t w=0, swId=0;
    while (w<W) {
        if (cwTiles[w] > config.splitTileThreshold) {
            std::uint32_t chunk=config.splitChunkTiles;
            std::uint32_t chunks=(cwTiles[w]+chunk-1)/chunk;
            for (std::uint32_t c=0; c<chunks; ++c) {
                std::uint32_t tb=cwFirst[w]+c*chunk;
                std::uint32_t te=std::min(cwLast[w], tb+chunk);
                sched.tasks.push_back(makeSw(w, w+1, swId, tb, te, SWT_SPLIT_CHUNK, sched, nullptr));
                ++swId; ++sched.splitChunks;
            }
            ++w; continue;
        }
        std::uint32_t cwBegin=w, end=w;
        std::vector<std::uint32_t> merged=cwRows[w];
        std::vector<unsigned> freq(merged.size(), 1);
        while (end+1<W) {
            std::uint32_t n=end+1;
            if (cwTiles[n] > config.splitTileThreshold) break;
            if (std::uint32_t(n-cwBegin+1) > config.maxCwsPerSw) break;
            std::uint64_t mergeTiles=0;
            for (auto i=cwBegin; i<=n; ++i) mergeTiles += cwTiles[i];
            if (mergeTiles > config.maxTilesPerTask) break;
            std::uint64_t overlap=0, occ=0;
            for (auto r : cwRows[n])
                if (std::binary_search(merged.begin(), merged.end(), r)) ++overlap;
            for (std::size_t i=0;i<merged.size();++i) occ += freq[i];
            occ += cwRows[n].size();
            if (double(overlap) < config.minReuseSavingFraction * double(occ)) break;
            for (auto r : cwRows[n]) {
                auto it=std::lower_bound(merged.begin(), merged.end(), r);
                if (it!=merged.end() && *it==r) ++freq[std::size_t(it-merged.begin())];
                else { auto pos=std::size_t(it-merged.begin()); merged.insert(merged.begin()+pos, r); freq.insert(freq.begin()+pos, 1); }
            }
            end=n;
        }
        sched.tasks.push_back(makeSw(cwBegin, end+1, swId, cwFirst[cwBegin], cwLast[end], 0, sched, nullptr));
        ++swId;
        w=end+1;
    }
    return sched;
}

ColumnAdaptiveFormat buildAdaptiveColumn(const ColumnBaseFormat& base, const ColumnSchedule& sched) {
    ColumnAdaptiveFormat out;
    out.rows=base.rows; out.cols=base.cols;
    out.colWindowOffset=base.colWindowOffset;
    out.tileLocalBit=base.tileLocalBit;
    out.tileValueOffset=base.tileValueOffset;
    out.values=base.values;
    out.swRowOffset=sched.swRowOffset;
    out.swRows=sched.swRows;
    out.swRowCacheSlot=sched.swRowCacheSlot;
    out.swCachedRowSlot=sched.swCachedRowSlot;
    out.swCachedCount=sched.swCachedCount;
    out.tasks=sched.tasks;
    out.tileRowSlot.assign(base.tileRows.size(), UINT32_MAX);
    for (auto& t : sched.tasks) {
        const auto* swRows=out.swRows.data()+t.rowBase;
        const std::uint32_t count=out.swRowOffset[t.swId+1]-out.swRowOffset[t.swId];
        for (auto tile=t.tileBegin; tile<t.tileEnd; ++tile) {
            for (unsigned r=0; r<8; ++r) {
                auto row=base.tileRows[std::size_t(tile)*8+r];
                if (row==UINT32_MAX) continue;
                auto it=std::lower_bound(swRows, swRows+count, row);
                if (it==swRows+count || *it!=row) throw std::runtime_error("task row missing from SW mapping");
                out.tileRowSlot[std::size_t(tile)*8+r]=std::uint32_t(it-swRows);
            }
        }
    }
    return out;
}
