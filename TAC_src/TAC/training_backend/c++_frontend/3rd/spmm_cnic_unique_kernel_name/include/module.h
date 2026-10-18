#pragma once

#include <torch/extension.h>
#include <vector>
#include <memory>
#include <iostream>
#include "class.h" 
#include "utils.h"
#include "spmm_launcher.h"

torch::Tensor forward_tensorcore_mixed(
    vint bcsr_num_rows, vint bcsr_num_cols, vint bcsr_nnz, vint bcsr_num_TCBlocks, vint bcsr_num_rowWindows,
    torch::Tensor bcsr_rowWindowOffset, torch::Tensor bcsr_tcOffset, torch::Tensor bcsr_sparseA2B, torch::Tensor bcsr_tcLocalBit, 
    vint bcsc_num_rows, vint bcsc_num_cols, vint bcsc_nnz, vint bcsc_num_TCBlocks, vint bcsc_num_colWindows,
    torch::Tensor bcsc_colWindowOffset, torch::Tensor bcsc_tcOffset, torch::Tensor bcsc_sparseA2C, torch::Tensor bcsc_tcLocalBit, 
    int64_t cpu_float_feature_len, int32_t gpu_node_capacity, int32_t feature_dim,
    torch::Tensor id_map, float* cpu_feature_cache_ptr, torch::Tensor cache_search_map, float** gpu_feature_cache_ptr,
    torch::Tensor x,   
    const int32_t device_id, float* denseC);

torch::Tensor forward_tensorcore(
    vint bcsr_num_rows, vint bcsr_num_cols, vint bcsr_nnz, vint bcsr_num_TCBlocks, vint bcsr_num_rowWindows,
    torch::Tensor bcsr_rowWindowOffset, torch::Tensor bcsr_tcOffset, torch::Tensor bcsr_sparseA2B, torch::Tensor bcsr_tcLocalBit, 
    vint bcsc_num_rows, vint bcsc_num_cols, vint bcsc_nnz, vint bcsc_num_TCBlocks, vint bcsc_num_colWindows,
    torch::Tensor bcsc_colWindowOffset, torch::Tensor bcsc_tcOffset, torch::Tensor bcsc_sparseA2C, torch::Tensor bcsc_tcLocalBit, 
    int32_t feature_dim, torch::Tensor x, const int32_t device_id, float* denseC);

torch::Tensor forward_tensorcore_backward(
    vint bcsr_num_rows, vint bcsr_num_cols, vint bcsr_nnz, vint bcsr_num_TCBlocks, vint bcsr_num_rowWindows,
    torch::Tensor bcsr_rowWindowOffset, torch::Tensor bcsr_tcOffset, torch::Tensor bcsr_sparseA2B, torch::Tensor bcsr_tcLocalBit, 
    vint bcsc_num_rows, vint bcsc_num_cols, vint bcsc_nnz, vint bcsc_num_TCBlocks, vint bcsc_num_colWindows,
    torch::Tensor bcsc_colWindowOffset, torch::Tensor bcsc_tcOffset, torch::Tensor bcsc_sparseA2C, torch::Tensor bcsc_tcLocalBit, 
    int32_t feature_dim, torch::Tensor x, const int32_t device_id, float* denseC);

void freeMatC();

void spmm_cnic_initialize(int device_id);
std::vector<void*> space_malloc(int device_id, vint denseC_size);
void spmm_cnic_finalize(int device_id);