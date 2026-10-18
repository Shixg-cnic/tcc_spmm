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
    src/kernel/column_adaptive.cu
    src/format/int4_quantize.cu
    src/kernel/column_fast.cu
    src/format/cw_order.cpp)
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
add_executable(int4_test tests/int4_test.cu)
target_compile_options(int4_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
target_link_libraries(int4_test PRIVATE tcc_spmm)
add_test(NAME int4_correctness COMMAND int4_test)
add_executable(compare_row_column tests/compare_row_column.cu)
target_compile_options(compare_row_column PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
target_link_libraries(compare_row_column PRIVATE tcc_spmm)

# Unified FP32 baseline benchmark: original Acc kernel + Row + Column C0..C3.
add_executable(benchmark_fp32 bench/benchmark_fp32.cu tests/acc_reference.cu)
target_include_directories(benchmark_fp32 PRIVATE ${CUSPARSE_INCLUDE_DIR} ${CUBLAS_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/tests)
target_compile_options(benchmark_fp32 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
target_link_libraries(benchmark_fp32 PRIVATE tcc_spmm)

# INT4 fused-kernel benchmark: quantised X + fused fixed/adaptive INT4 kernels
# vs the independent dequantize+FP32 baseline vs the FP32 pipeline.
add_executable(benchmark_int4 bench/benchmark_int4.cu)
target_include_directories(benchmark_int4 PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_compile_options(benchmark_int4 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
target_link_libraries(benchmark_int4 PRIVATE tcc_spmm)

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

# Output-path diagnosis (task-book batch 1): D0 reference C0, D1 no-writeback,
# D1b non-atomic store, D2 atomic-only same-address-stream, D3 offline analysis.
add_executable(column_output_diagnosis bench/column_output_diagnosis.cu)
target_include_directories(column_output_diagnosis PRIVATE ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/include)
target_compile_options(column_output_diagnosis PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
target_link_libraries(column_output_diagnosis PRIVATE tcc_spmm)

# Fast Column kernels (task-book batch 2): correctness + ablation benchmark.
add_executable(column_fast_test tests/column_fast_test.cu)
target_include_directories(column_fast_test PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_compile_options(column_fast_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
target_link_libraries(column_fast_test PRIVATE tcc_spmm)
add_test(NAME column_fast_correctness COMMAND column_fast_test)

# Batch 3, scheme A: bounded partial results + per-row reduction (two-phase).
add_executable(column_partial_reduce bench/column_partial_reduce.cu)
target_include_directories(column_partial_reduce PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_compile_options(column_partial_reduce PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
target_link_libraries(column_partial_reduce PRIVATE tcc_spmm)

# Batch 4: best FP32 Column kernels converted to fused INT4 X + the Acc Basic
# INT4 Fused baseline (mechanical copy of the Acc kernel with fused INT4 loads;
# the third_party original stays untouched; acc_int4_fused.cu is a single-TU
# Acc unit carrying converter + FP32 + INT4 launchers to avoid ODR clashes).
add_executable(benchmark_fast_int4 bench/benchmark_fast_int4.cu bench/acc_int4_fused.cu)
target_include_directories(benchmark_fast_int4 PRIVATE ${CMAKE_SOURCE_DIR}/tests
    ${CUSPARSE_INCLUDE_DIR} ${CUBLAS_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/third_party/Acc-SpMM/include)
target_compile_options(benchmark_fast_int4 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
target_link_libraries(benchmark_fast_int4 PRIVATE tcc_spmm)

# CW execution-order experiment (experiment/cw-order-only): offline orders +
# structure statistics, ordered V2C benchmark and correctness test.
add_executable(analyze_cw_order bench/analyze_cw_order.cpp)
target_include_directories(analyze_cw_order PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_compile_options(analyze_cw_order PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
target_link_libraries(analyze_cw_order PRIVATE tcc_spmm)

add_executable(benchmark_cw_order bench/benchmark_cw_order.cu tests/acc_reference.cu)
target_include_directories(benchmark_cw_order PRIVATE ${CMAKE_SOURCE_DIR}/tests
    ${CUSPARSE_INCLUDE_DIR} ${CUBLAS_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/third_party/Acc-SpMM/include)
target_compile_options(benchmark_cw_order PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
target_link_libraries(benchmark_cw_order PRIVATE tcc_spmm)

add_executable(cw_order_test tests/cw_order_test.cu tests/acc_reference.cu)
target_include_directories(cw_order_test PRIVATE ${CMAKE_SOURCE_DIR}/tests
    ${CUSPARSE_INCLUDE_DIR} ${CUBLAS_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/third_party/Acc-SpMM/include)
target_compile_options(cw_order_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
target_link_libraries(cw_order_test PRIVATE tcc_spmm)
add_test(NAME cw_order_correctness COMMAND cw_order_test)

# Current mainline: explicit source mapping + bounded MinHash grouping + Q4 lifetimes.
target_sources(tcc_spmm PRIVATE src/format/column_q4.cpp src/kernel/column_q4.cu)
add_executable(q4_test tests/q4_test.cu)
target_link_libraries(q4_test PRIVATE tcc_spmm)
add_test(NAME q4_correctness COMMAND q4_test)
add_executable(benchmark_q4 bench/benchmark_q4.cu tests/acc_reference.cu)
target_include_directories(benchmark_q4 PRIVATE tests ${CUSPARSE_INCLUDE_DIR} ${CUBLAS_INCLUDE_DIR})
target_compile_options(benchmark_q4 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
target_link_libraries(benchmark_q4 PRIVATE tcc_spmm)

add_executable(benchmark_q4_round2 bench/benchmark_q4_round2.cu tests/acc_reference.cu)
target_include_directories(benchmark_q4_round2 PRIVATE tests ${CUSPARSE_INCLUDE_DIR} ${CUBLAS_INCLUDE_DIR})
target_compile_options(benchmark_q4_round2 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
target_link_libraries(benchmark_q4_round2 PRIVATE tcc_spmm)

# Isolated TAC BCSC reproduction: no training/IPC or project SpMM kernels.
add_executable(tac_bcsc_repro repro/tac_bcsc/main.cu src/format/csr.cpp)
target_include_directories(tac_bcsc_repro PRIVATE tests include)
target_compile_options(tac_bcsc_repro PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
target_link_libraries(tac_bcsc_repro PRIVATE CUDA::cudart)
add_test(NAME tac_bcsc_correctness COMMAND tac_bcsc_repro --test)

add_executable(tac_bcsr_repro repro/tac_bcsr/main.cu src/format/csr.cpp)
target_include_directories(tac_bcsr_repro PRIVATE tests include)
target_compile_options(tac_bcsr_repro PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
target_link_libraries(tac_bcsr_repro PRIVATE CUDA::cudart)
add_test(NAME tac_bcsr_correctness COMMAND tac_bcsr_repro --test)

add_executable(benchmark_q4_datasets bench/benchmark_q4_datasets.cu tests/acc_reference.cu)
target_include_directories(benchmark_q4_datasets PRIVATE tests ${CUSPARSE_INCLUDE_DIR} ${CUBLAS_INCLUDE_DIR})
target_compile_options(benchmark_q4_datasets PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
target_link_libraries(benchmark_q4_datasets PRIVATE tcc_spmm)
