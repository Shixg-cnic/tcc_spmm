// Acc-SpMM memory/PTX helpers, MIT (c) 2025 Haisha Zhao.
#pragma once
#include <cuda_runtime.h>
#include <cstdint>
__device__ __forceinline__
void wait_group() {
    asm volatile(
        "cp.async.commit_group;\n"
        "cp.async.wait_group 0;\n"
        ::
    );
}

__device__ __forceinline__
void async_copy (std::uint32_t shared_addr, const float* val) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(shared_addr), "l"(val));
}

__device__ __forceinline__
void async_copy_idx (std::uint32_t shared_addr, const std::uint32_t* val) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(shared_addr), "l"(val));
}

__device__ __forceinline__
float load_fp32_from_global(const float* a) {
    float r;
    asm volatile("ld.global.ca.f32 %0, [%1];" : "=f"(r) : "l"(a));
    return r;
}

__device__ __forceinline__
float load_fp32_from_global2shared(const float* a) {
    float r;
    asm volatile("ld.global.cv.f32 %0, [%1];" : "=f"(r) : "l"(a));
    return r;
}

__device__ __forceinline__
std::uint32_t load_int_from_global(const std::uint32_t* a) {
    int r;
    asm volatile("ld.global.ca.s32 %0, [%1];" : "=r"(r) : "l"(a));      // s32:signed integer
    return r;
}

__device__ __forceinline__
void store_fp32_to_global(float* a, float v) {
    asm volatile("st.global.wt.f32 [%0], %1;" :: "l"(a), "f"(v));
}
