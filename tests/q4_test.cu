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
        unsigned owner[32];std::fill(owner,owner+32,UINT32_MAX);
        for(auto o=task.begin;o<task.end;++o) for(unsigned r=0;r<8;++r) {
            auto row=m.tileRows[m.tileOrder[o]*8+r];if(row!=UINT32_MAX)++total[row];
        }
        U+=total.size();
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
            }
            for(auto s:release)owner[s]=UINT32_MAX;
        }
        for(auto x:owner)require(x==UINT32_MAX,"unflushed row");
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
                for(auto v:{Q4Version::Q1,Q4Version::Sync,Q4Version::Bits,Q4Version::Float4,Q4Version::Pipeline}) {
                    check(launchQ4(gpu.view,dx.data,dy.data,F,v,stream));verify(error(dy.host(stream),expected));
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
        check(cudaStreamDestroy(stream));std::cout<<"Q4 structure, lifetimes, deterministic grouping, fallback and all GPU variants passed\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
