#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

// CUDA failures are reported here, rather than propagated through every API.
inline void checkCuda(cudaError_t result) {
    if (result != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(result));
}

// Own a device allocation; zero-sized workspaces do not allocate memory.
template <typename T> class DeviceBuffer {
  public:
    T* ptr = nullptr;

    explicit DeviceBuffer(std::size_t count) {
        if (count)
            checkCuda(cudaMalloc(&ptr, count * sizeof(T)));
    }

    explicit DeviceBuffer(const std::vector<T>& data) : DeviceBuffer(data.size()) {
        if (!data.empty())
            checkCuda(
                cudaMemcpy(ptr, data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice));
    }

    ~DeviceBuffer() {
        if (ptr)
            cudaFree(ptr);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};
