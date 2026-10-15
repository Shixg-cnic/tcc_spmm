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
    src/format/csr.cpp
    src/format/row_convert.cpp
    src/kernel/row_spmm.cu
    src/format/column_convert.cpp
    src/kernel/column_spmm.cu
    src/format/column_schedule.cpp
    src/kernel/column_adaptive.cu)
set(TCC_SW_SIZE 16 CACHE STRING "Default Column SuperWindow size")
set(TCC_CACHE_SIZE 32 CACHE STRING "Column cached rows per SuperWindow")
target_compile_definitions(tcc_spmm PUBLIC TCC_SW_SIZE=${TCC_SW_SIZE} TCC_CACHE_SIZE=${TCC_CACHE_SIZE})
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
add_executable(column_spmm_test tests/column_spmm_test.cu)
target_compile_options(column_spmm_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
target_link_libraries(column_spmm_test PRIVATE tcc_spmm)
add_test(NAME column_spmm_correctness COMMAND column_spmm_test)
add_executable(adaptive_sw_test tests/adaptive_sw_test.cu)
target_compile_options(adaptive_sw_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
target_link_libraries(adaptive_sw_test PRIVATE tcc_spmm)
add_test(NAME adaptive_sw_correctness COMMAND adaptive_sw_test)
add_executable(compare_row_column tests/compare_row_column.cu)
target_compile_options(compare_row_column PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
target_link_libraries(compare_row_column PRIVATE tcc_spmm)

# Unified FP32 baseline benchmark: original Acc kernel + Row + Column C0..C3.
add_executable(benchmark_fp32 bench/benchmark_fp32.cu tests/acc_reference.cu)
target_include_directories(benchmark_fp32 PRIVATE ${CUSPARSE_INCLUDE_DIR} ${CUBLAS_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/tests)
target_compile_options(benchmark_fp32 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
target_link_libraries(benchmark_fp32 PRIVATE tcc_spmm)

# Offline Column-SWTCF structural analysis (atomic/workload statistics).
add_executable(analyze_structure bench/analyze_structure.cu)
target_include_directories(analyze_structure PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_compile_options(analyze_structure PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
target_link_libraries(analyze_structure PRIVATE tcc_spmm)

# Diagnostic: C0 flow without global atomics, to isolate the atomic output cost.
add_executable(atomic_probe bench/atomic_probe.cu)
target_include_directories(atomic_probe PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_compile_options(atomic_probe PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
target_link_libraries(atomic_probe PRIVATE tcc_spmm)
