#include "format/csr.h"
#include "format/sw32.h"
#include "format/int8_features.cuh"
#include "spmm.cuh"

#include <cuda_runtime.h>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

template <typename T> class DeviceBuffer {
public:
    T* ptr = nullptr;
    explicit DeviceBuffer(std::size_t n) {
        if (n) check(cudaMalloc(&ptr, n * sizeof(T)));
    }
    explicit DeviceBuffer(const std::vector<T>& data) : DeviceBuffer(data.size()) {
        if (!data.empty()) check(cudaMemcpy(ptr, data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " matrix.mtx 128|256 gpu_id\n";
        return 1;
    }
    try {
        const unsigned features = std::stoul(argv[2]);
        const int device = std::stoi(argv[3]);
        if (features != 128 && features != 256) throw std::runtime_error("feature dimension must be 128 or 256");
        check(cudaSetDevice(device));
        cudaDeviceProp gpu{};
        check(cudaGetDeviceProperties(&gpu, device));
        std::cout << "GPU " << device << ": " << gpu.name << "\nReading Matrix Market: " << argv[1] << std::endl;
        auto matrix = loadMatrixMarket(argv[1]);
        const auto rows = matrix.rows, cols = matrix.cols;
        const auto nnz = matrix.csr.sourceIds.size();
        std::cout << "Converting " << rows << " x " << cols << ", nnz=" << nnz << " to SW32..." << std::endl;
        auto sw = convertCsrToSw32(rows, cols, matrix.csr);
        matrix.csr = CsrMatrix{};
        std::cout << "Uploading SW32 and preparing random features..." << std::endl;
        DeviceBuffer<std::uint32_t> offsets(sw.windowOffsets), sourceIds(sw.sourceIds),
            masks(sw.rowGroupMasks), buckets(sw.bucketOffsets), partialIds(sw.partialIds);
        DeviceBuffer<Sw32Task> tasks(sw.tasks);
        DeviceBuffer<Sw32ReduceRange> reductions(sw.reductions);
        DeviceBuffer<float> partials(std::size_t(sw.partialCount)*8*features);
        const auto counterCount = sw.partialCount ? (sw.windowOffsets.size()-1)*(features/128) : 0;
        DeviceBuffer<std::uint32_t> counters(counterCount);
        if (counterCount) check(cudaMemset(counters.ptr,0,counterCount*sizeof(std::uint32_t)));
        SpmmWorkspace workspace{partials.ptr,counters.ptr};
        DeviceBuffer<std::uint64_t> metadata(sw.sourceMetadata);
        DeviceBuffer<float> values(sw.values);
        Sw32DeviceView view{offsets.ptr, sourceIds.ptr, masks.ptr, metadata.ptr,
                            values.ptr, buckets.ptr, tasks.ptr, reductions.ptr,
                            partialIds.ptr,static_cast<std::uint32_t>(sw.tasks.size())};
        sw = Sw32HostFormat{};

        const std::size_t inputElements = std::size_t(cols) * features;
        std::vector<float> x(inputElements);
        std::mt19937 rng(20261025);
        for (float& v : x) v = float(rng() >> 8) * (1.0f / 8388608.0f) - 1.0f;
        DeviceBuffer<float> scales(cols), output(std::size_t(rows) * features);
        DeviceBuffer<std::int8_t> packed(inputElements);
        {
            DeviceBuffer<float> input(x);
            std::vector<float>().swap(x);
            DeviceBuffer<std::int8_t> q(inputElements);
            check(quantizeFeaturesInt8(input.ptr, q.ptr, scales.ptr, cols, features));
            check(packFeaturesInt8x16(q.ptr, packed.ptr, cols, features));
            check(cudaDeviceSynchronize());
        }
        std::cout << "Running fused INT8 dequantization + SpMM (F=" << features << ")..." << std::endl;
        auto launch = [&] { check(launchSpmmInt8(view, packed.ptr, scales.ptr, output.ptr, rows, cols, features, workspace)); };
        for (int i = 0; i < 10; ++i) launch();
        check(cudaDeviceSynchronize());
        cudaEvent_t start, end;
        check(cudaEventCreate(&start));
        check(cudaEventCreate(&end));
        check(cudaEventRecord(start));
        for (int i = 0; i < 10; ++i) launch();
        check(cudaEventRecord(end));
        check(cudaEventSynchronize(end));
        float ms = 0;
        check(cudaEventElapsedTime(&ms, start, end));
        cudaEventDestroy(start);
        cudaEventDestroy(end);
        float first[8];
        check(cudaMemcpy(first, output.ptr, sizeof(first), cudaMemcpyDeviceToHost));
        std::cout << std::fixed << std::setprecision(6)
                  << "SpMM kernel average: " << ms / 10 << " ms (10 launches; includes dequantization)\n"
                  << "Matrix read, format conversion, upload, quantization and packing are not timed.\n"
                  << "Output Y[0,0:8]:";
        for (float v : first) std::cout << ' ' << v;
        std::cout << '\n';
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
