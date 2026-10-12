cmake_minimum_required(VERSION 3.18)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "CUDA architectures")
endif()
project(TccTensorSpMM LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
find_package(CUDAToolkit REQUIRED)

# Conda may provide these headers through NVIDIA's Python packages.
file(GLOB nvidia_headers
    "$ENV{CONDA_PREFIX}/lib/python*/site-packages/nvidia/cusparse/include"
    "$ENV{CONDA_PREFIX}/lib/python*/site-packages/nvidia/cublas/include"
    "${CUDAToolkit_BIN_DIR}/../lib/python*/site-packages/nvidia/cusparse/include"
    "${CUDAToolkit_BIN_DIR}/../lib/python*/site-packages/nvidia/cublas/include")
find_path(CUSPARSE_INCLUDE_DIR cusparse.h HINTS ${CUDAToolkit_INCLUDE_DIRS} ${nvidia_headers})
find_path(CUBLAS_INCLUDE_DIR cublas_v2.h HINTS ${CUDAToolkit_INCLUDE_DIRS} ${nvidia_headers})
if(NOT CUSPARSE_INCLUDE_DIR OR NOT CUBLAS_INCLUDE_DIR)
    message(FATAL_ERROR "cuSPARSE and cuBLAS headers are required by the original Acc reference")
endif()
add_library(tcc_spmm STATIC
    src/format/row_convert.cpp
    src/kernel/row_spmm.cu)
target_include_directories(tcc_spmm PUBLIC include)
target_compile_options(tcc_spmm PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--ptxas-options=-v>)
target_link_libraries(tcc_spmm PUBLIC CUDA::cudart)

add_executable(row_spmm_test tests/row_spmm_test.cu tests/acc_reference.cu)
target_include_directories(row_spmm_test PRIVATE ${CUSPARSE_INCLUDE_DIR} ${CUBLAS_INCLUDE_DIR})
target_compile_options(row_spmm_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
target_link_libraries(row_spmm_test PRIVATE tcc_spmm)
enable_testing()
add_test(NAME row_spmm_correctness COMMAND row_spmm_test)
