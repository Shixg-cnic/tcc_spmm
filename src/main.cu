#include "cuda_support.cuh"
#include "device_matrix.cuh"
#include "format/int8_features.cuh"

#include <iomanip>
#include <iostream>
#include <random>
#include <string>

namespace {
std::vector<float> makeRandomFeatures(std::size_t count) {
    std::vector<float> features(count);
    std::mt19937 rng(20261025);
    for (float& value : features)
        value = float(rng() >> 8) * (1.0f / 8388608.0f) - 1.0f;
    return features;
}

void prepareFeatures(std::vector<float>& hostX, std::int8_t* packedX, float* scales,
                     unsigned sources, unsigned features) {
    const auto count = hostX.size();
    DeviceBuffer<float> input(hostX);
    std::vector<float>().swap(hostX);
    DeviceBuffer<std::int8_t> quantized(count);
    quantizeFeaturesInt8(input.ptr, quantized.ptr, scales, sources, features);
    packFeaturesInt8x16(quantized.ptr, packedX, sources, features);
    checkCuda(cudaDeviceSynchronize());
}

template <typename Launch> float benchmark(Launch launch) {
    constexpr int warmups = 10;
    constexpr int iterations = 10;
    for (int i = 0; i < warmups; ++i)
        launch();
    checkCuda(cudaDeviceSynchronize());

    cudaEvent_t start, end;
    checkCuda(cudaEventCreate(&start));
    checkCuda(cudaEventCreate(&end));
    checkCuda(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i)
        launch();
    checkCuda(cudaEventRecord(end));
    checkCuda(cudaEventSynchronize(end));

    float milliseconds = 0;
    checkCuda(cudaEventElapsedTime(&milliseconds, start, end));
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    return milliseconds / iterations;
}

void run(const std::string& path, unsigned features, int device) {
    checkCuda(cudaSetDevice(device));
    cudaDeviceProp gpu{};
    checkCuda(cudaGetDeviceProperties(&gpu, device));
    std::cout << "GPU " << device << ": " << gpu.name << "\nReading Matrix Market: " << path
              << std::endl;

    // 1. Read the matrix and build the SW32 format plus bounded panel tasks.
    auto matrix = loadMatrixMarket(path);
    const auto rows = matrix.rows, cols = matrix.cols;
    std::cout << "Converting " << rows << " x " << cols << ", nnz=" << matrix.csr.sourceIds.size()
              << " to SW32..." << std::endl;
    auto sw = convertCsrToSw32(rows, cols, matrix.csr);
    matrix.csr = CsrMatrix{};

    // 2. Upload A and its task plan; initialize the split-task completion counters.
    std::cout << "Uploading SW32 and preparing random features..." << std::endl;
    DeviceMatrix gpuMatrix(sw, features);
    sw = Sw32HostFormat{};
    const auto view = gpuMatrix.view();
    const auto workspace = gpuMatrix.workspace();

    // 3. Prepare INT8 X outside the timed region. SpMM dequantizes it in registers.
    const std::size_t inputElements = std::size_t(cols) * features;
    auto hostX = makeRandomFeatures(inputElements);
    DeviceBuffer<float> scales(cols), output(std::size_t(rows) * features);
    DeviceBuffer<std::int8_t> packedX(inputElements);
    prepareFeatures(hostX, packedX.ptr, scales.ptr, cols, features);

    // 4. Time the same fused kernel, including the in-kernel partial reduction.
    std::cout << "Running fused INT8 dequantization + SpMM (F=" << features << ")..." << std::endl;
    const float milliseconds = benchmark([&] {
        launchSpmmInt8(view, packedX.ptr, scales.ptr, output.ptr, rows, cols, features, workspace);
    });

    float first[8];
    checkCuda(cudaMemcpy(first, output.ptr, sizeof(first), cudaMemcpyDeviceToHost));
    std::cout << std::fixed << std::setprecision(6) << "SpMM kernel average: " << milliseconds
              << " ms (10 launches; includes dequantization)\n"
              << "Matrix read, format conversion, upload, quantization and packing are not timed.\n"
              << "Output Y[0,0:8]:";
    for (float value : first)
        std::cout << ' ' << value;
    std::cout << '\n';
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " matrix.mtx 128|256 gpu_id\n";
        return 1;
    }
    try {
        const unsigned features = std::stoul(argv[2]);
        if (features != 128 && features != 256)
            throw std::invalid_argument("feature dimension must be 128 or 256");
        run(argv[1], features, std::stoi(argv[3]));
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
