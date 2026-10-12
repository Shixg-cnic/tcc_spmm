#pragma once
#include "column_test_utils.cuh"

namespace test {
inline Matrix readMatrix(const std::string& path) {
    std::ifstream file(path,std::ios::binary); require(bool(file),"cannot open matrix");
    char magic[8];file.read(magic,8);require(file.gcount()==8,"truncated matrix header");file.seekg(0);
    if(std::memcmp(magic,"TCCCSR1\0",8)==0) {
        file.seekg(8);std::uint32_t header[3];file.read(reinterpret_cast<char*>(header),12);
        require(bool(file) && header[0] && header[0]<UINT32_MAX-7 && header[1] && header[1]<UINT32_MAX,"invalid CSR header");
        CanonicalCSR csr;csr.ptr.resize(std::size_t(header[0])+1);csr.src.resize(header[2]);csr.values.resize(header[2]);
        file.read(reinterpret_cast<char*>(csr.ptr.data()),csr.ptr.size()*4);
        file.read(reinterpret_cast<char*>(csr.src.data()),csr.src.size()*4);
        file.read(reinterpret_cast<char*>(csr.values.data()),csr.values.size()*4);
        require(bool(file) && file.peek()==EOF,"invalid CSR file size");
        return {header[0],header[1],canonicalCSR(header[0],header[1],header[2],csr.ptr.data(),csr.src.data(),csr.values.data())};
    }
    std::string line,banner,object,storage,type,symmetry;std::getline(file,line);
    std::istringstream h(line);h>>banner>>object>>storage>>type>>symmetry;
    require(banner=="%%MatrixMarket" && object=="matrix" && storage=="coordinate" &&
        (type=="real" || type=="integer" || type=="pattern") &&
        (symmetry=="general" || symmetry=="symmetric"),"unsupported MatrixMarket header");
    auto next=[&]() {
        while(std::getline(file,line)) if(!line.empty() && line[0]!='%') return;
        throw std::runtime_error("truncated MatrixMarket");
    };
    next();std::uint64_t rows,cols,nnz;std::istringstream shape(line);
    require(bool(shape>>rows>>cols>>nnz) && rows && cols && rows<UINT32_MAX-7 && cols<UINT32_MAX && nnz<UINT32_MAX &&
        (symmetry!="symmetric" || rows==cols),"invalid MatrixMarket shape");
    std::vector<std::vector<std::pair<unsigned,float>>> buckets(rows);
    for(std::uint64_t e=0;e<nnz;++e) {
        next();std::istringstream entry(line);std::uint64_t r,c;float v=1;
        require(bool(entry>>r>>c) && (type=="pattern" || bool(entry>>v)) && r && c && r<=rows && c<=cols && std::isfinite(v),"invalid entry");
        buckets[r-1].emplace_back(c-1,v);
        if(symmetry=="symmetric" && r!=c) buckets[c-1].emplace_back(r-1,v);
    }
    CanonicalCSR csr;
    for(auto& row:buckets) {
        for(auto item:row) {csr.src.push_back(item.first);csr.values.push_back(item.second);}
        require(csr.src.size()<UINT32_MAX,"CSR index overflow");csr.ptr.push_back(csr.src.size());
    }
    return {unsigned(rows),unsigned(cols),canonicalCSR(rows,cols,csr.src.size(),csr.ptr.data(),csr.src.data(),csr.values.data())};
}
inline std::vector<float> readFeatures(const char* path,unsigned cols,unsigned f) {
    if(!path) return features(cols,f);
    std::ifstream file(path,std::ios::binary|std::ios::ate);
    require(bool(file),"cannot open X"); auto bytes=file.tellg();
    require(bytes>0 && std::uint64_t(bytes)%(std::uint64_t(cols)*4)==0,"X must contain K*sourceF FP32 values");
    auto width=std::uint64_t(bytes)/(std::uint64_t(cols)*4);
    std::vector<float> out(std::size_t(cols)*f),row(width);
    file.seekg(0);
    for(unsigned r=0;r<cols;++r) {
        file.read(reinterpret_cast<char*>(row.data()),width*4);
        for(unsigned j=0;j<f;++j) {require(std::isfinite(row[j%width]),"nonfinite X");out[std::size_t(r)*f+j]=row[j%width];}
    }
    require(bool(file),"truncated X");return out;
}
}
