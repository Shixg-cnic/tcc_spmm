#pragma once
#include <cuda_runtime.h>
#include <cstdint>
using vint=uint32_t;
using MAT_PTR_TYPE=uint32_t;
using MAT_VAL_TYPE=float;
using TCLOCAL_TYPE=uint64_t;
constexpr unsigned COL_WINDOW_R=16;
// Same RNA TF32 rounding and MMA as TAC; separate output registers fix the
// upstream helper's integer-typed FP32 sources and writes to input-only operands.
__device__ __forceinline__ unsigned rounded_tf32(float value) {
    unsigned bits;
    asm("cvt.rna.tf32.f32 %0, %1;":"=r"(bits):"f"(value));
    return bits;
}
__device__ __forceinline__ void tf32_m16n8k8(float* a,float* b,float* c) {
    unsigned a0=rounded_tf32(a[0]),a1=rounded_tf32(a[1]),a2=rounded_tf32(a[2]),a3=rounded_tf32(a[3]);
    unsigned b0=rounded_tf32(b[0]),b1=rounded_tf32(b[1]);
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}
__device__ __forceinline__ void wait_group() {
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;" ::: "memory");
}
__device__ __forceinline__ void async_copy_idx(unsigned shared_addr,const unsigned* value) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(shared_addr),"l"(value):"memory");
}
__device__ __forceinline__ float load_fp32_from_global(const float* ptr) {
    float value;asm volatile("ld.global.ca.f32 %0, [%1];":"=f"(value):"l"(ptr):"memory");return value;
}
__device__ __forceinline__ unsigned load_int_from_global(const unsigned* ptr) {
    unsigned value;asm volatile("ld.global.cv.u32 %0, [%1];":"=r"(value):"l"(ptr):"memory");return value;
}
