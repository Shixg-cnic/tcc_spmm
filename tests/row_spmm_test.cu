#include "row_format.cuh"
#include "acc_reference.cuh"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>

namespace {
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

template<class T> struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(const std::vector<T>& host) {
        if (!host.empty()) {
            check(cudaMalloc(reinterpret_cast<void**>(&data), host.size()*sizeof(T)));
            check(cudaMemcpy(data, host.data(), host.size()*sizeof(T), cudaMemcpyHostToDevice));
        }
    }
    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

float tf32(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    bits = (bits + 4096u) & 0xffffe000u;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

template<class Launch> float measure(Launch launch, int repeats, cudaStream_t stream) {
    for (int i=0; i<10; ++i) launch();
    check(cudaStreamSynchronize(stream));
    cudaGraph_t graph;
    cudaGraphExec_t execution;
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    for (int i=0; i<repeats; ++i) launch();
    check(cudaStreamEndCapture(stream, &graph));
    check(cudaGraphInstantiate(&execution, graph, nullptr, nullptr, 0));
    cudaEvent_t begin, end;
    check(cudaEventCreate(&begin)); check(cudaEventCreate(&end));
    std::vector<float> samples;
    for (int i=0; i<5; ++i) {
        check(cudaEventRecord(begin, stream));
        check(cudaGraphLaunch(execution, stream));
        check(cudaEventRecord(end, stream)); check(cudaEventSynchronize(end));
        float ms;
        check(cudaEventElapsedTime(&ms, begin, end));
        samples.push_back(ms / repeats);
    }
    cudaEventDestroy(begin); cudaEventDestroy(end);
    cudaGraphExecDestroy(execution); cudaGraphDestroy(graph);
    std::sort(samples.begin(), samples.end());
    return samples[2];
}

struct Matrix {
    std::uint32_t rows, cols;
    CanonicalCSR csr;
};

Matrix readMatrix(const char* path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("cannot open matrix");
    std::string line, banner, object, storage, type, symmetry;
    std::getline(file, line);
    std::istringstream header(line);
    header >> banner >> object >> storage >> type >> symmetry;
    if (banner != "%%MatrixMarket" || object != "matrix" || storage != "coordinate" ||
        (type != "real" && type != "integer" && type != "pattern") ||
        (symmetry != "general" && symmetry != "symmetric"))
        throw std::runtime_error("expected real/integer/pattern coordinate MatrixMarket");
    auto nextLine = [&]() {
        while (std::getline(file, line)) if (!line.empty() && line[0] != '%') return;
        throw std::runtime_error("truncated matrix");
    };
    nextLine();
    std::uint64_t rows, cols, edges;
    std::istringstream size(line);
    if (!(size >> rows >> cols >> edges) || !rows || !cols || rows > UINT32_MAX-7 || cols >= UINT32_MAX ||
        edges > UINT32_MAX || (symmetry == "symmetric" && rows != cols))
        throw std::runtime_error("invalid matrix dimensions");
    std::vector<std::vector<std::pair<std::uint32_t,float>>> buckets(rows);
    for (std::uint64_t i=0; i<edges; ++i) {
        nextLine();
        std::istringstream entry(line);
        std::uint64_t row, col;
        float value=1.f;
        if (!(entry >> row >> col) || (type != "pattern" && !(entry >> value)) ||
            !row || row > rows || !col || col > cols || !std::isfinite(value))
            throw std::runtime_error("invalid matrix entry");
        buckets[row-1].emplace_back(col-1,value);
        if (symmetry == "symmetric" && row != col) buckets[col-1].emplace_back(row-1,value);
    }
    CanonicalCSR raw;
    for (const auto& bucket : buckets) {
        for (auto entry : bucket) { raw.src.push_back(entry.first); raw.values.push_back(entry.second); }
        if (raw.src.size() > UINT32_MAX) throw std::runtime_error("too many entries");
        raw.ptr.push_back(raw.src.size());
    }
    return {static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(cols),
        canonicalCSR(rows,cols,raw.src.size(),raw.ptr.data(),raw.src.data(),raw.values.data())};
}

void run(const Matrix& matrix, unsigned features, int repeats, cudaStream_t stream, const char* featurePath=nullptr) {
    const auto& csr = matrix.csr;
    auto begin = std::chrono::steady_clock::now();
    auto host = convertRow(matrix.rows,matrix.cols,csr);
    double convertMs = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
    auto original = convertAccRow(matrix.rows,matrix.cols,csr);
    if (host.rowWindowOffset != original.rowWindowOffset || host.tcOffset != original.tcOffset ||
        host.sparseA2B != original.sparseA2B || host.tcLocalBit != original.tcLocalBit || host.values != original.values)
        throw std::runtime_error("format differs from original Acc");
    std::vector<float> x(static_cast<std::size_t>(matrix.cols)*features);
    if (featurePath) {
        std::ifstream file(featurePath,std::ios::binary|std::ios::ate);
        if (!file || file.tellg() != std::streamoff(x.size()*sizeof(float)))
            throw std::runtime_error("X binary must contain exactly K*F float32 values");
        file.seekg(0); file.read(reinterpret_cast<char*>(x.data()), x.size()*sizeof(float));
        if (!file) throw std::runtime_error("failed to read X");
    } else {
        std::mt19937 generator(17);
        std::uniform_real_distribution<float> distribution(-1,1);
        for (auto& value : x) value=distribution(generator);
    }
    auto roundedX=x;
    for (auto& value : roundedX) {
        if (!std::isfinite(value)) throw std::runtime_error("X must be finite");
        value=tf32(value);
    }
    std::vector<double> expected(static_cast<std::size_t>(matrix.rows)*features,0.);
    for (unsigned row=0; row<matrix.rows; ++row)
        for (auto e=csr.ptr[row]; e<csr.ptr[row+1]; ++e)
            for (unsigned f=0; f<features; ++f)
                expected[static_cast<std::size_t>(row)*features+f] +=
                    double(tf32(csr.values[e])) * roundedX[static_cast<std::size_t>(csr.src[e])*features+f];
    std::vector<float> output(expected.size(),std::numeric_limits<float>::quiet_NaN());
    std::vector<float> originalOutput(static_cast<std::size_t>((matrix.rows+7)/8*8)*features);
    DeviceBuffer<std::uint32_t> window(host.rowWindowOffset), offset(host.tcOffset), mapping(host.sparseA2B);
    DeviceBuffer<std::uint64_t> bitmap(host.tcLocalBit);
    DeviceBuffer<float> values(host.values), input(x), result(output), accResult(originalOutput);
    check(cudaDeviceSynchronize());
    auto launch = [&]() { check(launchRowSpMM(window.data,offset.data,mapping.data,bitmap.data,values.data,
        input.data,result.data,matrix.rows,matrix.cols,features,stream)); };
    launch(); check(cudaStreamSynchronize(stream));
    check(cudaMemcpy(output.data(),result.data,output.size()*sizeof(float),cudaMemcpyDeviceToHost));
    double squaredError=0, squaredReference=0, maxError=0, maxReference=0;
    for (std::size_t i=0; i<output.size(); ++i) {
        double diff=output[i]-expected[i];
        squaredError+=diff*diff; squaredReference+=expected[i]*expected[i];
        maxError=std::max(maxError,std::abs(diff)); maxReference=std::max(maxReference,std::abs(expected[i]));
    }
    double rmse=std::sqrt(squaredError/std::max(squaredReference,1e-24));
    if (!std::isfinite(rmse) || rmse > 1e-4 || maxError > 1e-4 + maxReference*5e-4)
        throw std::runtime_error("TF32 mathematical reference mismatch");
    float rowMs=measure(launch,repeats,stream);
    std::cout << "M=" << matrix.rows << " K=" << matrix.cols << " F=" << features
              << " nnz=" << csr.src.size() << " convert_cpu_ms=" << convertMs
              << " row_ms=" << rowMs << " relative_rmse_tf32=" << rmse;
    bool emptyWindow=false;
    for (std::size_t i=1; i<host.rowWindowOffset.size(); ++i)
        emptyWindow |= host.rowWindowOffset[i]==host.rowWindowOffset[i-1];
    if (!emptyWindow && features>=64 && features<=512 && features%32==0) {
        auto accLaunch = [&]() { check(launchAccRowSpMM(window.data,offset.data,mapping.data,bitmap.data,values.data,
            input.data,accResult.data,(matrix.rows+7)/8*8,matrix.cols,features,stream)); };
        accLaunch(); check(cudaStreamSynchronize(stream));
        check(cudaMemcpy(originalOutput.data(),accResult.data,originalOutput.size()*sizeof(float),cudaMemcpyDeviceToHost));
        if (!std::equal(output.begin(),output.end(),originalOutput.begin()))
            throw std::runtime_error("output differs from original Acc Row");
        std::cout << " acc_row_ms=" << measure(accLaunch,repeats,stream);
    }
    std::cout << '\n';
}

void selfTest(cudaStream_t stream) {
    for (unsigned rows : {16u,19u}) {
        CanonicalCSR raw;
        std::mt19937 generator(21);
        std::uniform_real_distribution<float> distribution(-1,1);
        for (unsigned row=0; row<rows; ++row) {
            for (unsigned col=0; col<37; ++col)
                if ((row<8 && col<8) || (row+1==rows && col==36) || distribution(generator)>.4f) {
                    raw.src.push_back(col); raw.values.push_back(distribution(generator));
                }
            raw.ptr.push_back(raw.src.size());
        }
        Matrix matrix{rows,37,canonicalCSR(rows,37,raw.src.size(),raw.ptr.data(),raw.src.data(),raw.values.data())};
        for (unsigned f : {1u,17u,32u,64u,100u,128u,256u,512u,513u,544u,1024u}) run(matrix,f,10,stream);
    }
    std::vector<std::uint32_t> ptr(18,4), src{2,0,2,0}; ptr[0]=0;
    std::vector<float> values{1,.25,-1,.5};
    auto duplicate=canonicalCSR(17,3,4,ptr.data(),src.data(),values.data());
    if (duplicate.src.size()!=1 || duplicate.values[0]!=.75f) throw std::runtime_error("duplicate coalescing failed");
    run({17,3,duplicate},100,10,stream);
    CanonicalCSR empty; empty.ptr.resize(10,0);
    run({9,3,empty},100,10,stream);
    try {
        std::uint32_t badPtr[]{0,1}, badSrc[]{5}; float weight[]{1};
        canonicalCSR(1,2,1,badPtr,badSrc,weight);
        throw std::runtime_error("invalid CSR accepted");
    } catch (const std::invalid_argument&) {}
    std::cout << "All native correctness tests passed.\n";
}
}

int main(int argc, char** argv) {
    try {
        if (argc>1 && std::string(argv[1])=="--help") {
            std::cout << "Usage: row_spmm_test [matrix.mtx [F=128 [repeat=100 [device=0 [X.bin]]]]]\n"
                      << "No arguments: run correctness tests. X.bin: row-major FP32 [K,F].\n";
            return 0;
        }
        if (argc>6) throw std::invalid_argument("too many arguments; use --help");
        int features=argc>2 ? std::stoi(argv[2]) : 128;
        int repeats=argc>3 ? std::stoi(argv[3]) : 100;
        if (features<1 || repeats<1) throw std::invalid_argument("F and repeat must be positive");
        check(cudaSetDevice(argc>4 ? std::stoi(argv[4]) : 0));
        cudaStream_t stream; check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        if (argc==1) selfTest(stream);
        else run(readMatrix(argv[1]),features,repeats,stream,argc>5 ? argv[5] : nullptr);
        check(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n'; return 1;
    }
}
