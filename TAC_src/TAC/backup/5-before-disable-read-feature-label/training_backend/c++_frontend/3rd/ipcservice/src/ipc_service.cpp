#include <fcntl.h>
#include <iostream>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>
#include <torch/extension.h>
#include "ipc_service.h"
#include "config.h"

namespace py = pybind11;

IPCEnv* env;
int32_t h_node_counter[16];
int32_t h_edge_counter[16];

void InitializeIPC(int gpu_number){
    env = NewIPCEnv();
    env->Initialize(gpu_number);
}

void FinalizeIPC(){
    env->Finalize();
}

std::vector<torch::Tensor> cuda_get_next(
    int32_t* ids,
    float* float_features, 
    int32_t* labels,
    int feature_dim,
    int32_t* agg_src,
    int32_t* agg_dst,
    int32_t* node_counter,
    int32_t* edge_counter,
    int32_t* h_node_counter,
    int32_t* h_edge_counter
    );

void waitSampleReady(int device_id);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


std::vector<torch::Tensor> get_next(int feature_dim) {
    env->Wait();
    int32_t* ids = env->GetIds();
    float* float_features = env->GetFloatFeatures();
    int32_t* labels = env->GetLabels();
    int32_t* agg_src = env->GetAggSrc();
    int32_t* agg_dst = env->GetAggDst();
    int32_t* node_counter = env->GetNodeCounter();
    int32_t* edge_counter = env->GetEdgeCounter();
    auto result = cuda_get_next(ids, float_features, labels,
                                feature_dim,
                                agg_src, agg_dst, 
                                node_counter, edge_counter, 
                                h_node_counter, h_edge_counter);
    return result;
}

std::vector<int> get_block_size() {
    std::vector<int> ret;
    int hop_num = h_node_counter[INTRABATCH_CON * 3 - 1];

    for(int i = hop_num; i > 0; i--){
        ret.push_back(h_node_counter[INTRABATCH_CON * 3 + i]);
        ret.push_back(h_node_counter[INTRABATCH_CON * 3 + i - 1]);
    }
    // int block1_src_node = h_node_counter[9];
    // int block1_dst_node = h_node_counter[7];
    // int block2_src_node = h_node_counter[7];
    // int block2_dst_node = h_node_counter[5];

    // ret.push_back(block1_src_node);
    // ret.push_back(block1_dst_node);
    // ret.push_back(block2_src_node);
    // ret.push_back(block2_dst_node);
    return ret;
}

std::vector<int32_t> get_steps(){
    std::vector<int32_t> ret;
    ret.push_back(env->GetTrainStep());
    ret.push_back(env->GetValidStep());
    ret.push_back(env->GetTestStep());
    return ret;
} 

void Synchronize(){
    env->Post();
}

std::vector<vint> get_sparseMatrix_size() {
    vint *sparseMatrixSize = env->GetSparseMatrixSize();

    std::vector<vint> ret; 
    for (int i = 0; i < 10; i++) {
        ret.push_back(sparseMatrixSize[i]);
    }

    return ret;
}

std::vector<torch::Tensor> get_sparseMatrix_data() {
    vint **bcsr = env->GetBCSRData();
    vint **bcsc = env->GetBCSCData();
    vint *sparseMatrixSize = env->GetSparseMatrixSize();

    std::vector<torch::Tensor> ret;

    int current_dev = -1;
    cudaGetDevice(&current_dev);
    auto device = "cuda:" + std::to_string(current_dev);
    //cudaCheckError();

    torch::Dtype dtypes[4] = {torch::kUInt32, torch::kUInt32, torch::kUInt32, torch::kUInt64};

    // (batch + 1hop + 2hop) * （batch + 1hop + 2hop) 图数据获取
    vint bcsr_num_TCBlocks = sparseMatrixSize[3];
    vint bcsr_num_rowWindows = sparseMatrixSize[4];

    vint bcsc_num_TCBlocks = sparseMatrixSize[8];
    vint bcsc_num_colWindows = sparseMatrixSize[9];

    // bcsr
    long long bcsr_length[4] = {bcsr_num_rowWindows + 1, bcsr_num_TCBlocks + 1, bcsr_num_TCBlocks * COL_WINDOW, bcsr_num_TCBlocks};
    for (int32_t i = 0; i < 4; i++) {
        torch::Tensor tensor = torch::from_blob(
            bcsr[i],
            {bcsr_length[i]},
            torch::TensorOptions().dtype(dtypes[i]).device(device));
          
        ret.push_back(tensor);
    }

    // bcsc
    long long bcsc_length[4] = {bcsc_num_colWindows + 1, bcsc_num_TCBlocks + 1, bcsc_num_TCBlocks * COL_WINDOW, bcsc_num_TCBlocks};
    for (int32_t i = 0; i < 4; i++) {
        torch::Tensor tensor = torch::from_blob(
            bcsc[i],
            {bcsc_length[i]},
            torch::TensorOptions().dtype(dtypes[i]).device(device));
          
        ret.push_back(tensor);
    }
    
    return ret;
}

std::vector<torch::Tensor> get_memoryaccess_data() {
    int32_t* cache_search_map = env->GetCacheSearchMap();
    vint *sparseMatrixSize = env->GetSparseMatrixSize();

    vint num_nodes = sparseMatrixSize[11]; // num_node = bcsr_num_rows = bcsc_num_rows(s_bcsr_num_cols, 全部采样节点数量)

    std::vector<torch::Tensor> ret;

    int current_dev = -1;
    cudaGetDevice(&current_dev);
    auto device = "cuda:" + std::to_string(current_dev);

    torch::Tensor cache_search_map_tensor = torch::from_blob(
        cache_search_map,
        {(long long)num_nodes},
        torch::TensorOptions().dtype(torch::kInt32).device(device));
      
    ret.push_back(cache_search_map_tensor);

    return ret;
}

std::vector<int64_t> get_feature_attribute() {
    int64_t* featureAttribute = env->GetFeatureAttribute();

    std::vector<int64_t> ret;
    
    for (int i = 0; i < 3; i++) {
        ret.push_back(featureAttribute[i]);
    }

    return ret;
}

std::vector<void*> get_feature_cache_ptr() {
    float* cpu_feature_cache = env->GetCpuFeatureCache();
    float** gpu_feature_cache = env->GetGpuFeatureCache();

    std::vector<void*> ret;

    ret.push_back(static_cast<void*>(cpu_feature_cache));
    ret.push_back(static_cast<void*>(gpu_feature_cache));

    return ret;
}

float simulate_feature_cache_lookup(
    int32_t batch_size,
    int32_t total_num_nodes,
    int32_t float_feature_len,
    int32_t cache_capacity,
    int32_t* sampled_ids,
    int32_t* cache_index, 
    float* cpu_float_features, 
    float** gpu_float_feature
);

// 模拟采样过程中缓存 feature 的过程
// 不采用这里实际提取的数据，只是模拟一遍访存过程，后续 mm 计算仍然使用原始 feature 接口返回的数据
float simulate_CacheLookupOP() {
    vint *sparseMatrixSize = env->GetSparseMatrixSize();
    int64_t* featureAttribute = env->GetFeatureAttribute();

    int32_t* ids = env->GetIds();
    int32_t* cache_search_map = env->GetCacheSearchMap();
    float* cpu_feature_cache = env->GetCpuFeatureCache();
    float** gpu_feature_cache = env->GetGpuFeatureCache(); 

    vint num_nodes = sparseMatrixSize[11]; 
    int64_t cpu_float_feature_len = featureAttribute[0];
    int64_t gpu_node_capacity = featureAttribute[1];
    int64_t feature_dim = featureAttribute[2];

    float elapsed = simulate_feature_cache_lookup(
        num_nodes,
        num_nodes,
        feature_dim,
        gpu_node_capacity,
        ids,
        cache_search_map,
        cpu_feature_cache,
        gpu_feature_cache
    );

    return elapsed;
}

void wait_sample_ready(int device_id) {
    // 为了使用 nvtx 显示执行阶段，所以只能下探到 cu 文件中的 env 来实现
    waitSampleReady(device_id);
}
std::vector<vint> get_data_scale(int device_id) {
    std::vector<vint> ret;

    // TODO 需要基于预采样过程评估得到真实数据
    // first_layer:     0:6556000   1:6694400   2:6696000   3:6566400   4:6708000   5:6696000   6:6627200   7:6802400
    // second_layer:    0:16783360  1:17137664  2:17141760  3:16809984  4:17172480  5:17141760  6:16965632  7:17414144
    vint denseC_size = 18000000;

    ret.push_back(denseC_size);

    return ret;
}
