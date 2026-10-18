// Acc-SpMM PTX MMA helper, MIT (c) 2025 Haisha Zhao.
#pragma once
#include <cuda_runtime.h>
#include <cstdint>
__device__ __forceinline__
void tf32_m16n8k8(float* MatA, float* MatB, float* MatC) {
    std::uint32_t const* A   = reinterpret_cast<std::uint32_t const*>(MatA);
    std::uint32_t const* B   = reinterpret_cast<std::uint32_t const*>(MatB);
    float* C        = reinterpret_cast<float*>(MatC);

    asm volatile(
        "cvt.rna.tf32.f32 %4, %4;\n"
        "cvt.rna.tf32.f32 %5, %5;\n"
        "cvt.rna.tf32.f32 %6, %6;\n"
        "cvt.rna.tf32.f32 %7, %7;\n"
        "cvt.rna.tf32.f32 %8, %8;\n"
        "cvt.rna.tf32.f32 %9, %9;\n"
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"
        "{%0, %1, %2, %3},"
        "{%4, %5, %6, %7},"
        "{%8, %9},"
        "{%0, %1, %2, %3};\n"
        :"+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3])      // output
        :"r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
         "r"(B[0]), "r"(B[1])
    );
}

__device__ __forceinline__ std::uint32_t toTf32RnaBits(float value) {
    std::uint32_t bits;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(bits) : "f"(value));
    return bits;
}
__device__ __forceinline__ void mmaTf32Bits(const std::uint32_t* a,const std::uint32_t* b,float* c) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
        : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
}
