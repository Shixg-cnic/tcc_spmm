#include "column_q4.cuh"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

namespace {
using U=unsigned;
using Clock=std::chrono::steady_clock;
double ms(Clock::time_point t) { return std::chrono::duration<double,std::milli>(Clock::now()-t).count(); }
U checked(std::size_t n) { if(n>=UINT32_MAX) throw std::overflow_error("Q4 index overflow"); return U(n); }
std::uint64_t hashRow(std::uint64_t x) {
    x+=0x9e3779b97f4a7c15ULL+20261018ULL;
    x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;
    x=(x^(x>>27))*0x94d049bb133111ebULL;
    return x^(x>>31);
}
}
Q4Format buildQ4(U rows,U cols,const CanonicalCSR& csr,bool grouped,std::uint64_t budget) {
    if(!rows||!cols||csr.ptr.size()!=std::size_t(rows)+1||csr.ptr[0]||
       csr.ptr.back()!=csr.src.size()||csr.values.size()!=csr.src.size())
        throw std::invalid_argument("Q4 CSR shape");
    checked(csr.src.size());
    auto start=Clock::now();
    std::vector<U> ptr(std::size_t(cols)+1,0);
    for(U r=0;r<rows;++r) {
        if(csr.ptr[r]>csr.ptr[r+1]||csr.ptr[r+1]>csr.src.size()) throw std::invalid_argument("Q4 CSR offsets");
        for(U e=csr.ptr[r];e<csr.ptr[r+1];++e) {
            U c=csr.src[e];
            if(c>=cols||!std::isfinite(csr.values[e])||csr.values[e]==0||
                (e>csr.ptr[r]&&csr.src[e-1]>=c)) throw std::invalid_argument("Q4 requires canonical CSR");
            ++ptr[c+1];
        }
    }
    std::partial_sum(ptr.begin(),ptr.end(),ptr.begin());
    auto cursor=ptr;
    std::vector<U> nbr(csr.src.size());
    for(U r=0;r<rows;++r) for(U e=csr.ptr[r];e<csr.ptr[r+1];++e) nbr[cursor[csr.src[e]]++]=r;
    std::vector<U> order(cols); std::iota(order.begin(),order.end(),0);
    auto degree=[&](U c){return ptr[c+1]-ptr[c];};
    Q4Format m; m.rows=rows;m.cols=cols;
    if(grouped) {
        std::vector<std::uint64_t> sig(cols,UINT64_MAX);
        for(U c=0;c<cols;++c) for(U e=ptr[c];e<ptr[c+1];++e) sig[c]=std::min(sig[c],hashRow(nbr[e]));
        std::sort(order.begin(),order.end(),[&](U a,U b){
            if(bool(degree(a))!=bool(degree(b))) return degree(a)>0;
            if(sig[a]!=sig[b]) return sig[a]<sig[b];
            return degree(a)!=degree(b)?degree(a)<degree(b):a<b;
        });
        for(std::size_t base=0;base<cols;base+=32) {
            U n=std::min<std::size_t>(32,cols-base); std::uint64_t edges=0;
            for(U i=0;i<n;++i) edges+=degree(order[base+i]);
            if(edges>budget) {++m.fallbackGroups;continue;}
            std::unordered_map<U,U> masks; masks.reserve(edges);
            for(U i=0;i<n;++i) for(U e=ptr[order[base+i]];e<ptr[order[base+i]+1];++e) masks[nbr[e]]|=U(1)<<i;
            std::vector<std::vector<U>> lists(n);
            std::vector<U> membership; membership.reserve(masks.size());
            for(auto kv:masks) {
                U id=checked(membership.size());membership.push_back(kv.second);
                for(U bits=kv.second;bits;bits&=bits-1) lists[__builtin_ctz(bits)].push_back(id);
            }
            U used=0; std::vector<U> result;
            for(U cw=0;cw<n;cw+=8) {
                std::vector<bool> covered(membership.size(),false);
                std::array<U,32> gain{};
                for(U j=0;j<std::min(8u,n-cw);++j) {
                    U best=UINT32_MAX;
                    for(U i=0;i<n;++i) if(!(used&(U(1)<<i))) {
                        U c=order[base+i];
                        if(best==UINT32_MAX||gain[i]>gain[best]||
                           (gain[i]==gain[best]&&(degree(c)>degree(order[base+best])||
                            (degree(c)==degree(order[base+best])&&c<order[base+best])))) best=i;
                    }
                    used|=U(1)<<best;result.push_back(order[base+best]);
                    for(U id:lists[best]) if(!covered[id]) {
                        covered[id]=true;
                        for(U bits=membership[id]&~used;bits;bits&=bits-1) ++gain[__builtin_ctz(bits)];
                    }
                }
            }
            std::copy(result.begin(),result.end(),order.begin()+base);
        }
    }
    // Sorting each CW makes bitmap columns follow original IDs, including fallback CWs.
    for(std::size_t b=0;b<cols;b+=8) std::sort(order.begin()+b,order.begin()+std::min<std::size_t>(cols,b+8));
    m.cwCols=order; m.cwCols.resize((std::size_t(cols)+7)/8*8,UINT32_MAX);
    m.groupingMs=ms(start);start=Clock::now();
    struct Edge {U row,col;float value;};
    U W=checked(m.cwCols.size()/8);
    std::vector<U> inverse(cols),offset(std::size_t(W)+1,0);
    for(U i=0;i<cols;++i) inverse[order[i]]=i;
    for(U c:csr.src) ++offset[inverse[c]/8+1];
    std::partial_sum(offset.begin(),offset.end(),offset.begin());cursor=offset;
    std::vector<Edge> edges(csr.src.size());
    // Original CSR columns sorted, and each CW sorted by original ID: already bitmap order.
    for(U r=0;r<rows;++r) for(U e=csr.ptr[r];e<csr.ptr[r+1];++e) {
        U p=inverse[csr.src[e]];edges[cursor[p/8]++]={r,p%8,csr.values[e]};
    }
    for(U w=0;w<W;++w) {
        U nr=0;std::uint64_t bits=0;
        for(U e=offset[w];e<offset[w+1];++e) {
            if(e==offset[w]||edges[e].row!=edges[e-1].row) {
                if(nr==8) {m.bitmap.push_back(bits);m.valueOffset.push_back(checked(m.values.size()));nr=0;bits=0;}
                m.tileRows.push_back(edges[e].row);++nr;++m.P;
            }
            bits|=1ULL<<((nr-1)*8+edges[e].col);m.values.push_back(edges[e].value);
        }
        if(nr) {
            while(nr++<8) m.tileRows.push_back(UINT32_MAX);
            m.bitmap.push_back(bits);m.valueOffset.push_back(checked(m.values.size()));
        }
        m.cwOffset.push_back(checked(m.bitmap.size()));
    }
    checked(m.tileRows.size());m.formatMs=ms(start);start=Clock::now();
    for(U first=0;first<W;first+=4) {
        Q4Task task{};task.begin=checked(m.tileOrder.size());task.count=std::min(4u,W-first);
        U head[4]{};
        for(U c=0;c<task.count;++c) {task.cw[c]=first+c;head[c]=m.cwOffset[first+c];}
        auto maximum=[&](U t){U max=0;for(U r=0;r<8;++r) if(m.tileRows[std::size_t(t)*8+r]!=UINT32_MAX) max=m.tileRows[std::size_t(t)*8+r];return max;};
        while(true) {
            U best=UINT32_MAX;
            for(U c=0;c<task.count;++c) if(head[c]<m.cwOffset[first+c+1])
                if(best==UINT32_MAX||maximum(head[c])<maximum(head[best])) best=c;
            if(best==UINT32_MAX) break;
            m.tileOrder.push_back(head[best]++);m.orderCW.push_back(best);
        }
        task.end=checked(m.tileOrder.size());
        std::unordered_map<U,U> remaining,active;
        for(U o=task.begin;o<task.end;++o) for(U r=0;r<8;++r) {
            U row=m.tileRows[std::size_t(m.tileOrder[o])*8+r];if(row!=UINT32_MAX) ++remaining[row];
        }
        m.U+=remaining.size();U occupied=0;
        for(U o=task.begin;o<task.end;++o) {
            U release=0;
            for(U r=0;r<8;++r) {
                U row=m.tileRows[std::size_t(m.tileOrder[o])*8+r];
                if(row==UINT32_MAX) {m.rowToken.push_back(Q4_INVALID);continue;}
                auto it=active.find(row);bool prev=it!=active.end(),next=--remaining[row]>0;
                U slot=0;
                if(prev) slot=it->second;
                else if(next) {
                    if(occupied==UINT32_MAX) throw std::runtime_error("Q4 exceeds 32 live slots");
                    slot=__builtin_ctz(~occupied);occupied|=U(1)<<slot;active[row]=slot;
                    m.peakSlots=std::max(m.peakSlots,U(__builtin_popcount(occupied)));
                }
                m.rowToken.push_back(slot|(prev?Q4_PREV:0)|(next?Q4_NEXT:0));
                if(prev&&!next) {release|=U(1)<<slot;active.erase(row);}
            }
            occupied&=~release; // Freed slots are reusable only in the next tile.
        }
        if(occupied||!active.empty()) throw std::runtime_error("Q4 unfinished lifetime");
        m.tasks.push_back(task);
    }
    m.planMs=ms(start);return m;
}
