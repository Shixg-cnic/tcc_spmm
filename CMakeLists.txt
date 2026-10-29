cmake_minimum_required(VERSION 3.18)
set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "CUDA architecture")
project(TccSpMM LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
find_package(CUDAToolkit REQUIRED)

add_library(tcc_spmm STATIC
    src/format/matrix_market.cpp
    src/format/csr.cpp
    src/format/sw32.cpp
    src/format/int8_quantize.cu
    src/format/pack_int8.cu
    src/kernel/spmm.cu)
target_include_directories(tcc_spmm PUBLIC include)
target_compile_options(tcc_spmm PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo>)
target_link_libraries(tcc_spmm PUBLIC CUDA::cudart)

add_executable(tcc_spmm_run src/main.cu)
target_link_libraries(tcc_spmm_run PRIVATE tcc_spmm)
