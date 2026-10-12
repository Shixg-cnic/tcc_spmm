#include "column_format.cuh"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {
std::uint32_t checked(std::size_t n) {
    if (n >= UINT32_MAX) throw std::overflow_error("Column-SWTCF 32-bit index overflow");
    return static_cast<std::uint32_t>(n);
}
struct Edge { std::uint32_t row, col; float value; };
}
ColumnSWTCF convertColumn(std::uint32_t rows, std::uint32_t cols,
    const CanonicalCSR& csr, std::uint32_t swSize) {
    if (!rows || !cols || !swSize || csr.ptr.size() != std::size_t(rows)+1 ||
        csr.ptr.front() || csr.ptr.back() != csr.src.size() || csr.values.size() != csr.src.size())
        throw std::invalid_argument("invalid Column CSR dimensions/offsets");
    checked(csr.src.size());
    const std::uint32_t windows = (std::uint64_t(cols)+7)/8;
    const std::uint32_t supers = (std::uint64_t(windows)+swSize-1)/swSize;
    checked(std::size_t(supers)*COLUMN_CACHE_SIZE);
    std::vector<std::size_t> offsets(std::size_t(windows)+1,0);
    for (std::uint32_t row=0; row<rows; ++row) {
        if (csr.ptr[row] > csr.ptr[row+1] || csr.ptr[row+1] > csr.src.size())
            throw std::invalid_argument("invalid CSR offset");
        for (auto e=csr.ptr[row]; e<csr.ptr[row+1]; ++e) {
            if (csr.src[e] >= cols || !std::isfinite(csr.values[e]) || csr.values[e] == 0 ||
                (e>csr.ptr[row] && csr.src[e]<=csr.src[e-1]))
                throw std::invalid_argument("convertColumn requires canonical CSR");
            ++offsets[csr.src[e]/8+1];
        }
    }
    for (std::uint32_t w=0; w<windows; ++w) offsets[w+1]+=offsets[w];
    auto cursor=offsets;
    std::vector<Edge> edges(csr.src.size());
    // CSR traversal preserves ascending target row and local column order.
    for (std::uint32_t row=0; row<rows; ++row)
        for (auto e=csr.ptr[row]; e<csr.ptr[row+1]; ++e)
            edges[cursor[csr.src[e]/8]++] = {row,csr.src[e]%8,csr.values[e]};
    ColumnSWTCF out;
    out.rows=rows; out.cols=cols; out.superWindowSize=swSize;
    for (std::uint32_t s=0; s<supers; ++s) {
        const auto first=std::uint64_t(s)*swSize;
        const auto end=std::min<std::uint64_t>(first+swSize,windows);
        std::vector<std::uint32_t> occurrences;
        for (auto w=first; w<end; ++w)
            for (auto e=offsets[w]; e<offsets[w+1]; ++e)
                if (e==offsets[w] || edges[e].row!=edges[e-1].row) occurrences.push_back(edges[e].row);
        std::sort(occurrences.begin(),occurrences.end());
        std::vector<std::uint32_t> swRows, frequency;
        for (auto row : occurrences) {
            if (swRows.empty() || swRows.back()!=row) { swRows.push_back(row); frequency.push_back(1); }
            else ++frequency.back();
        }
        checked(out.superWindowRows.size()+swRows.size());
        std::vector<std::uint32_t> candidates;
        for (std::size_t i=0; i<swRows.size(); ++i) if (frequency[i]>1) candidates.push_back(checked(i));
        std::sort(candidates.begin(),candidates.end(),[&](auto a, auto b) {
            return frequency[a]!=frequency[b] ? frequency[a]>frequency[b] : swRows[a]<swRows[b];
        });
        const auto count=std::min<std::size_t>(COLUMN_CACHE_SIZE,candidates.size());
        std::vector<std::uint32_t> slots(swRows.size(),UINT32_MAX);
        for (std::size_t c=0; c<COLUMN_CACHE_SIZE; ++c) {
            auto rowSlot=c<count ? candidates[c] : UINT32_MAX;
            out.superWindowCachedRowSlot.push_back(rowSlot);
            if (c<count) slots[rowSlot]=c;
        }
        out.superWindowCachedCount.push_back(count);
        out.superWindowRows.insert(out.superWindowRows.end(),swRows.begin(),swRows.end());
        out.superWindowRowCacheSlot.insert(out.superWindowRowCacheSlot.end(),slots.begin(),slots.end());
        out.superWindowRowOffset.push_back(checked(out.superWindowRows.size()));
        for (auto w=first; w<end; ++w) {
            std::uint64_t bitmap=0;
            std::uint32_t localRow=0;
            for (auto e=offsets[w]; e<offsets[w+1]; ++e) {
                if (e==offsets[w] || edges[e].row!=edges[e-1].row) {
                    if (localRow==8) {
                        out.tileLocalBit.push_back(bitmap); out.tileValueOffset.push_back(checked(out.values.size()));
                        bitmap=0; localRow=0;
                    }
                    checked(out.tileRowSlot.size()+1);
                    auto pos=std::lower_bound(swRows.begin(),swRows.end(),edges[e].row)-swRows.begin();
                    out.tileRowSlot.push_back(checked(pos)); ++localRow;
                }
                bitmap |= std::uint64_t{1} << ((localRow-1)*8+edges[e].col);
                out.values.push_back(edges[e].value);
            }
            if (localRow) {
                checked(out.tileRowSlot.size()+8-localRow);
                while (localRow++<8) out.tileRowSlot.push_back(UINT32_MAX);
                out.tileLocalBit.push_back(bitmap); out.tileValueOffset.push_back(checked(out.values.size()));
            }
            out.colWindowOffset.push_back(checked(out.tileLocalBit.size()));
        }
    }
    return out;
}
