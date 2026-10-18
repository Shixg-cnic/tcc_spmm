#include "q4_utils.cuh"
#include <map>
#include <set>
using namespace test;
void validate(const Q4Format& m,const Matrix& input) {
    std::vector<std::map<unsigned,float>> decoded(m.rows);
    std::set<unsigned> sources;
    for(auto c:m.cwCols) if(c!=UINT32_MAX) require(c<m.cols&&sources.insert(c).second,"source permutation");
    require(sources.size()==m.cols,"missing source");
    for(unsigned w=0;w+1<m.cwOffset.size();++w) {
        unsigned last=0;bool first=true;
        for(auto t=m.cwOffset[w];t<m.cwOffset[w+1];++t) {
            unsigned e=m.valueOffset[t];
            for(unsigned r=0;r<8;++r) {
                auto row=m.tileRows[t*8+r];if(row==UINT32_MAX)continue;
                require(first||row>last,"CW rows not sorted/unique");first=false;last=row;
            }
            for(unsigned b=0;b<64;++b) if(m.bitmap[t]&(1ULL<<b)) {
                auto row=m.tileRows[t*8+b/8],c=m.cwCols[w*8+b%8];
                require(row<m.rows&&c<m.cols,"padding nonzero");
                require(decoded[row].emplace(c,m.values[e++]).second,"duplicate coordinate");
            }
            require(e==m.valueOffset[t+1],"bitmap values");
        }
    }
    for(unsigned r=0;r<m.rows;++r) {
        require(decoded[r].size()==input.csr.ptr[r+1]-input.csr.ptr[r],"row size");
        for(auto e=input.csr.ptr[r];e<input.csr.ptr[r+1];++e) require(decoded[r].at(input.csr.src[e])==input.csr.values[e],"weight changed");
    }
    std::vector<unsigned> seen(m.bitmap.size(),0);std::uint64_t P=0,U=0;
    for(auto task:m.tasks) {
        std::map<unsigned,unsigned> total,visits,slot;
        unsigned peak=0;
        unsigned owner[32];std::fill(owner,owner+32,UINT32_MAX);
        for(auto o=task.begin;o<task.end;++o) for(unsigned r=0;r<8;++r) {
            auto row=m.tileRows[m.tileOrder[o]*8+r];if(row!=UINT32_MAX)++total[row];
        }
        U+=total.size();
        std::uint64_t taskP=0;for(auto kv:total)taskP+=kv.second;
        auto stats=m.taskStats.at(task.cw[0]/4);
        require(stats.P==taskP&&stats.U==total.size(),"per-task P/U statistics");
        unsigned heads[4];for(unsigned c=0;c<task.count;++c)heads[c]=m.cwOffset[task.cw[c]];
        auto maxRow=[&](unsigned t){unsigned x=0;for(unsigned r=0;r<8;++r)if(m.tileRows[t*8+r]!=UINT32_MAX)x=m.tileRows[t*8+r];return x;};
        for(auto o=task.begin;o<task.end;++o) {
            unsigned t=m.tileOrder[o],cw=m.orderCW[o];require(cw<task.count,"local CW");
            require(t==heads[cw]++,"CW order");
            for(unsigned c=0;c<task.count;++c) if(c!=cw&&heads[c]<m.cwOffset[task.cw[c]+1]) require(maxRow(t)<=maxRow(heads[c]),"not tileMax merge");
            require(++seen.at(t)==1,"tile repeated");std::vector<unsigned> release;
            for(unsigned r=0;r<8;++r) {
                auto row=m.tileRows[t*8+r];unsigned token=m.rowToken[o*8+r],s=token&31;
                if(row==UINT32_MAX){require(token==Q4_INVALID,"padding token");continue;}
                ++P;bool prev=visits[row]++>0,next=visits[row]<total[row];
                require(bool(token&Q4_PREV)==prev&&bool(token&Q4_NEXT)==next,"lifetime flags");
                if(prev) require(owner[s]==row&&slot[row]==s,"cache ownership");
                else if(next){require(owner[s]==UINT32_MAX,"same-tile slot reuse");owner[s]=row;slot[row]=s;}
                if(prev&&!next)release.push_back(s);
                unsigned live=0;for(auto value:owner)live+=value!=UINT32_MAX;
                peak=std::max(peak,live);
            }
            for(auto s:release)owner[s]=UINT32_MAX;
        }
        for(auto x:owner)require(x==UINT32_MAX,"unflushed row");
        require(peak==stats.peakSlots,"per-task peak statistics");
    }
    for(auto n:seen)require(n==1,"missing tile");require(P==m.P&&U==m.U,"statistics");
}
int main() {
    try {
        cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        std::vector<Matrix> matrices;
        matrices.push_back(makeMatrix(9,5,[](unsigned,unsigned){return 0.f;}));
        matrices.push_back(makeMatrix(49,67,[](unsigned r,unsigned c){return (r*17+c*11)%13==0?float(int(c%9)-4)*.13f:0.f;}));
        matrices.push_back(makeMatrix(113,65,[](unsigned r,unsigned c){return (r+c)%7<4?float(int((r+c)%11)-5)*.07f:0.f;}));
        matrices.push_back(makeMatrix(257,127,[](unsigned r,unsigned c){return c%8==0||r%17==0?float(int((r*3+c)%7)-3)*.19f:0.f;}));
        matrices.push_back(makeMatrix(63,32,[](unsigned r,unsigned c){return r/9==c/8||r%5==0?1.f:0.f;}));
        for(auto& input:matrices) for(bool grouped:{false,true}) for(auto budget:{0ULL,262144ULL}) {
            auto m=buildQ4(input.rows,input.cols,input.csr,grouped,budget);validate(m,input);
            auto again=buildQ4(input.rows,input.cols,input.csr,grouped,budget);
            require(m.cwCols==again.cwCols&&m.tileOrder==again.tileOrder&&m.rowToken==again.rowToken,"nondeterministic");
            if(grouped&&budget==0&&!input.csr.src.empty())require(m.fallbackGroups>0,"fallback untested");
            GPUQ4 gpu(m);
            for(unsigned F:{128,256}) {
                auto x=features(m.cols,F);auto expected=reference(input,x,F,true);
                Buffer<float> dx(x),dy(std::size_t(m.rows)*F);
                for(auto v:{Q4Version::Q1,Q4Version::Sync,Q4Version::Bits,Q4Version::Float4,Q4Version::Pipeline,Q4Version::ScalarPipeline}) {
                    for(unsigned h:{16u,24u,32u}) if(h>=m.peakSlots) for(bool xorA:{false,true}) {
                        check(launchQ4(gpu.view,dx.data,dy.data,F,Q4Config{v,h,xorA},stream));
                        verify(error(dy.host(stream),expected));
                        ColumnKernelInfo info;check(q4Attributes(Q4Config{v,h,xorA},m.peakSlots,&info));
                        if(v!=Q4Version::Q1) require(info.sharedBytes==h*132*4+(v==Q4Version::Pipeline||v==Q4Version::ScalarPipeline?512:256),"wrong H specialization");
                    }
                }
            }
        }
        // Randomized structure and weighted numerical coverage with partial CW/SW tails.
        std::mt19937 rng(20261018);
        for(unsigned it=0;it<40;++it) {
            auto input=makeMatrix(1+rng()%130,1+rng()%100,[&](unsigned,unsigned){return rng()%8==0?float(int(rng()%200)-100)/113.f:0.f;});
            auto m=buildQ4(input.rows,input.cols,input.csr,true);validate(m,input);GPUQ4 gpu(m);
            auto x=features(m.cols,128);Buffer<float> dx(x),dy(m.rows*128);
            check(launchQ4(gpu.view,dx.data,dy.data,128,Q4Version::Pipeline,stream));verify(error(dy.host(stream),reference(input,x,128,true)));
        }
        // Enumerate actual logical shared addresses, not just bank indices.
        for(bool xorA:{false,true}) {
            std::set<unsigned> physical;
            for(unsigned t=0;t<64;++t)physical.insert(xorA?t^((t>>3)&4):t);
            require(physical.size()==64&&*physical.rbegin()==63,"A layout not bijective");
            for(unsigned part=0;part<2;++part) {
                std::map<unsigned,std::set<unsigned>> banks;
                for(unsigned lane=0;lane<32;++lane) {
                    unsigned g=lane/4,k=lane%4+4*part;
                    unsigned address=8*g+(k^(xorA?(g&4):0));banks[address%32].insert(address);
                }
                require(banks.size()==(xorA?32:16),"A bank count");
                for(auto bank:banks)require(bank.second.size()==(xorA?1:2),"A bank multiplicity");
            }
            for(unsigned warp=0;warp<2;++warp) {
                std::set<unsigned> banks;
                for(unsigned lane=0;lane<32;++lane){unsigned t=warp*32+lane; banks.insert((xorA?t^((t>>3)&4):t)%32);}
                require(banks.size()==32,"staging bank conflict");
            }
        }
        // Force live slots above 15/23 to catch accidental token & (H-1).
        auto dense=makeMatrix(8,32,[](unsigned r,unsigned c){return float((r+c)%5+1)/7;});
        for(unsigned shift:{16u,24u}) {
            auto m=buildQ4(dense.rows,dense.cols,dense.csr,false);
            require(m.peakSlots==8,"dense lifetime peak");
            unsigned h24Slots[8]={0,1,2,3,8,9,16,17};
            for(auto& token:m.rowToken)if(token&(Q4_PREV|Q4_NEXT))
                token=(token&~31)|(shift==16?h24Slots[token&31]:(token&31)+shift);
            m.peakSlots=shift+8;validateQ4Cache(m,q4CacheSlots(m.peakSlots));GPUQ4 gpu(m);
            auto x=features(m.cols,256);Buffer<float> dx(x),dy(m.rows*256);auto expected=reference(dense,x,256,true);
            for(auto v:{Q4Version::Bits,Q4Version::Pipeline,Q4Version::ScalarPipeline})for(bool xorA:{false,true}) {
                check(launchQ4(gpu.view,dx.data,dy.data,256,Q4Config{v,0,xorA},stream));verify(error(dy.host(stream),expected));
            }
            require(launchQ4(gpu.view,dx.data,dy.data,256,Q4Config{Q4Version::Bits,16,false},stream)==cudaErrorInvalidValue,"undersized H accepted");
            bool rejected=false;try{validateQ4Cache(m,16);}catch(const std::invalid_argument&){rejected=true;}require(rejected,"host H check");
        }
        // More than one 256-task bucket window, including a partial last window.
        auto input=makeMatrix(41,9001,[](unsigned r,unsigned c){return (c*7+r*3)%23==0?0.17f:0.f;});
        auto m=buildQ4(input.rows,input.cols,input.csr,true);auto original=m.tasks;
        m.tasks=reorderQ4Tasks(original);auto again=reorderQ4Tasks(original);
        require(m.tasks.size()==original.size(),"task count changed");
        unsigned changed=0;
        for(std::size_t b=0;b<original.size();b+=256) {
            std::multiset<std::vector<unsigned>> before,after;
            for(auto i=b;i<std::min(b+256,original.size());++i) {
                auto key=[](Q4Task t){return std::vector<unsigned>{t.begin,t.end,t.cw[0],t.cw[1],t.cw[2],t.cw[3],t.count};};
                before.insert(key(original[i]));after.insert(key(m.tasks[i]));
                require(key(m.tasks[i])==key(again[i]),"task shuffle nondeterministic");
                changed+=key(original[i])!=key(m.tasks[i]);
                auto bucket=[](Q4Task t){unsigned n=t.end-t.begin;return n?32-__builtin_clz(n):0;};
                require(bucket(original[i])==bucket(m.tasks[i]),"task workload bucket changed");
            }
            require(before==after,"task not moved intact/local");
        }
        require(changed>0,"task shuffle was identity");validate(m,input);
        GPUQ4 gpu(m);
        for(unsigned f:{128,256}) {
            auto x=features(m.cols,f);Buffer<float> dx(x),dy(m.rows*f);auto expected=reference(input,x,f,true);
            for(auto v:{Q4Version::Bits,Q4Version::Pipeline,Q4Version::ScalarPipeline}) {
                check(launchQ4(gpu.view,dx.data,dy.data,f,Q4Config{v,0,true},stream));verify(error(dy.host(stream),expected));
            }
        }
        check(cudaStreamDestroy(stream));std::cout<<"Q4 structure, lifetimes, deterministic grouping, fallback all H/layout/async/output variants, bank enumeration and task shuffle passed\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
