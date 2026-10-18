#include <fcntl.h>
#include <iostream>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>
#include "ipc_service.h"

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
    env->IpcCloseMemHandle();
    env->Post();
}

std::vector<vint> get_sparseMatrix_size() {
    vint *sparseMatrixSize = env->GetSparseMatrixSize();

    std::vector<vint> ret; 
    for (int i = 0; i < 20; i++) {
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

    torch::Dtype dtypes[5] = {torch::kUInt32, torch::kUInt32, torch::kUInt32, torch::kUInt64, torch::kFloat32};

    // batch * (batch + 1hop) 图数据获取
    vint bcsr_nnz = sparseMatrixSize[2];
    vint bcsr_num_TCBlocks = sparseMatrixSize[3];
    vint bcsr_num_rowWindows = sparseMatrixSize[4];

    vint bcsc_nnz = sparseMatrixSize[7];
    vint bcsc_num_TCBlocks = sparseMatrixSize[8];
    vint bcsc_num_colWindows = sparseMatrixSize[9];

    // f_bcsr
    long long f_bcsr_length[5] = {bcsr_num_rowWindows + 1, bcsr_num_TCBlocks + 1, bcsr_num_TCBlocks * COL_WINDOW, bcsr_num_TCBlocks, bcsr_nnz};
    for (int32_t i = 0; i < 5; i++) {
        torch::Tensor tensor = torch::from_blob(
            bcsr[i],
            {f_bcsr_length[i]},
            torch::TensorOptions().dtype(dtypes[i]).device(device));
          
        ret.push_back(tensor);
    }
    // f_bcsc
    long long f_bcsc_length[5] = {bcsc_num_colWindows + 1, bcsc_num_TCBlocks + 1, bcsc_num_TCBlocks * COL_WINDOW, bcsc_num_TCBlocks, bcsc_nnz};
    for (int32_t i = 0; i < 5; i++) {
        torch::Tensor tensor = torch::from_blob(
            bcsc[i],
            {f_bcsc_length[i]},
            torch::TensorOptions().dtype(dtypes[i]).device(device));
          
        ret.push_back(tensor);
    }
    
    // batch * （batch + 1hop + 2hop) 图数据获取
    bcsr_nnz = sparseMatrixSize[12];
    bcsr_num_TCBlocks = sparseMatrixSize[13];
    bcsr_num_rowWindows = sparseMatrixSize[14];

    bcsc_nnz = sparseMatrixSize[17];
    bcsc_num_TCBlocks = sparseMatrixSize[18];
    bcsc_num_colWindows = sparseMatrixSize[19];

    // s_bcsr
    long long s_bcsr_length[5] = {bcsr_num_rowWindows + 1, bcsr_num_TCBlocks + 1, bcsr_num_TCBlocks * COL_WINDOW, bcsr_num_TCBlocks, bcsr_nnz};
    for (int32_t i = 0; i < 5; i++) {
        torch::Tensor tensor = torch::from_blob(
            bcsr[5 + i],
            {s_bcsr_length[i]},
            torch::TensorOptions().dtype(dtypes[i]).device(device));
          
        ret.push_back(tensor);
    }
    // s_bcsc
    long long s_bcsc_length[5] = {bcsc_num_colWindows + 1, bcsc_num_TCBlocks + 1, bcsc_num_TCBlocks * COL_WINDOW, bcsc_num_TCBlocks, bcsc_nnz};
    for (int32_t i = 0; i < 5; i++) {
        torch::Tensor tensor = torch::from_blob(
            bcsc[5 + i],
            {s_bcsc_length[i]},
            torch::TensorOptions().dtype(dtypes[i]).device(device));
          
        ret.push_back(tensor);
    }
    
    return ret;
}

std::vector<torch::Tensor> get_memoryaccess_data() {
    int32_t* cache_search_map = env->GetCacheSearchMap();
    vint *sparseMatrixSize = env->GetSparseMatrixSize();

    vint num_nodes = sparseMatrixSize[0]; // num_node = bcsr_num_rows = bcsc_num_rows

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

// debug
void* get_host_cpu_feature_cache_ptr() {
    return env->GetHostCpuFeatureCache();
}
// debug