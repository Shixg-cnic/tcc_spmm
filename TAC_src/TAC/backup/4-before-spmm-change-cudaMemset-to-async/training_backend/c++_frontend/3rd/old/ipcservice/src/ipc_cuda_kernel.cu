#include <iostream>
#include <vector>
#include "../include/helper_multiprocess.h"
#include <stdio.h>
#include <stdlib.h>

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "ipc_service.h"

#include <thrust/random/uniform_int_distribution.h>
#include <thrust/random/linear_congruential_engine.h>

#define MAX_DEVICE 8
#define MEMORY_USAGE 8 // 完整的 cache_index 增加 1 个

// Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError()                                       \
  {                                                            \
    cudaError_t e = cudaGetLastError();                        \
    if (e != cudaSuccess) {                                    \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                           \
      exit(EXIT_FAILURE);                                      \
    }                                                          \
  }

__global__
void reconstruct_gpu_feature_cache(float** all_gpu_float_features, float** gpu_float_features) {
  int tid = threadIdx.x;
  //printf("thread-%d, all_gpu_float_features: {%p}, gpu_float_features: {%p}\n", tid, all_gpu_float_features[0], gpu_float_features[0]);
  all_gpu_float_features[tid] = gpu_float_features[tid];
}

  typedef struct shmStruct_st {
    int32_t steps[3];
    uint32_t sparseMatrixMetric[MAX_DEVICE][INTERBATCH_CON][20];
    int64_t featureAttribute[3];
    cudaIpcMemHandle_t memHandle[MAX_DEVICE][INTERBATCH_CON][MEMORY_USAGE];
    cudaIpcMemHandle_t sparseMatrixData[MAX_DEVICE][INTERBATCH_CON][20];
    cudaIpcMemHandle_t gpu_feature_memHandle[MAX_DEVICE];
  } shmStruct;

class GPUIPCEnv : public IPCEnv {
public: 
  int Initialize(int gpu_number) override {
    //volatile shmStruct *shm = NULL;
    //int central_device = -1;
    central_device = -1;
    cudaGetDevice(&central_device);
    cudaCheckError();
    std::printf("[Info] device %d ipc initialize\n", central_device);

    sharedMemoryInfo info;
    const char shmName[] = "fuse_simpleIPCshm";
    if (sharedMemoryCreate(shmName, sizeof(*shm), &info) != 0) {
      printf("Failed to create shared memory slab\n");
      exit(EXIT_FAILURE);
    }
    shm = (volatile shmStruct *)info.addr;

    // 利用 IpcHandle 读取 step 数据
    train_step_ = shm->steps[0];
    valid_step_ = shm->steps[1];
    test_step_ = shm->steps[2];
    ids_.resize(INTERBATCH_CON);
    float_features_.resize(INTERBATCH_CON);
    labels_.resize(INTERBATCH_CON);
    agg_src_.resize(INTERBATCH_CON);
    agg_dst_.resize(INTERBATCH_CON);
    node_counter_.resize(INTERBATCH_CON);
    edge_counter_.resize(INTERBATCH_CON);

    // 获取数组数据的 IpcHandle
    for(int i = 0; i < INTERBATCH_CON; i++){
      cudaIpcOpenMemHandle(&ids_[i], *(cudaIpcMemHandle_t*)&shm->memHandle[central_device][i][0], cudaIpcMemLazyEnablePeerAccess);
      cudaIpcOpenMemHandle(&float_features_[i], *(cudaIpcMemHandle_t*)&shm->memHandle[central_device][i][1], cudaIpcMemLazyEnablePeerAccess);
      cudaIpcOpenMemHandle(&labels_[i], *(cudaIpcMemHandle_t*)&shm->memHandle[central_device][i][2], cudaIpcMemLazyEnablePeerAccess);
      cudaIpcOpenMemHandle(&agg_src_[i], *(cudaIpcMemHandle_t*)&shm->memHandle[central_device][i][3], cudaIpcMemLazyEnablePeerAccess);
      cudaIpcOpenMemHandle(&agg_dst_[i], *(cudaIpcMemHandle_t*)&shm->memHandle[central_device][i][4], cudaIpcMemLazyEnablePeerAccess);
      cudaIpcOpenMemHandle(&node_counter_[i], *(cudaIpcMemHandle_t*)&shm->memHandle[central_device][i][5], cudaIpcMemLazyEnablePeerAccess);
      cudaIpcOpenMemHandle(&edge_counter_[i], *(cudaIpcMemHandle_t*)&shm->memHandle[central_device][i][6], cudaIpcMemLazyEnablePeerAccess);
    }

    // 获取混合数据结构以及 feature 相关的 IpcHandle
    initialize_dataStructure_feature(gpu_number);

    std::cout<<"CUDA: "<<central_device<<" IPC shared memory opened\n";

    semr_.resize(INTERBATCH_CON);
    semw_.resize(INTERBATCH_CON);
    for(int i = 0; i < INTERBATCH_CON; i++){
      std::string ssr = "fuse_sem_r_";
      std::string ssw = "fuse_sem_w_";
      std::string ssri = ssr + std::to_string(central_device) + "_" + std::to_string(i);
      std::string sswi = ssw + std::to_string(central_device) + "_" + std::to_string(i);
      semr_[i] = sem_open(ssri.c_str(), O_CREAT | O_RDWR, 0666, 0);
      if (semr_[i] == SEM_FAILED ){
        printf("errno = %d\n", errno );
        return -1;
      }
      semw_[i] = sem_open(sswi.c_str(), O_CREAT | O_RDWR, 0666, 0);
      if (semw_[i] == SEM_FAILED){
        printf("errno = %d\n", errno );
        return -1;
      }
      sem_post(semr_[i]);
    }

    current_pipe_ = 0;
    return central_device;
  }

  void Wait() override {
    sem_t* sem = semw_[current_pipe_];
    sem_wait(sem);
  }

  void Post() override {
    sem_t* sem = semr_[current_pipe_];
    sem_post(sem);
    current_pipe_ = (current_pipe_ + 1)%INTERBATCH_CON;
  }

  int32_t* GetIds() override {
    return (int32_t*)ids_[current_pipe_];
  }
  float* GetFloatFeatures() override {
    return (float*)float_features_[current_pipe_];
  }
  int32_t* GetLabels() override {
    return (int32_t*)labels_[current_pipe_];
  }
  int32_t* GetAggSrc() override {
    return (int32_t*)agg_src_[current_pipe_];
  }
  int32_t* GetAggDst() override {
    return (int32_t*)agg_dst_[current_pipe_];
  }
  int32_t* GetNodeCounter() override {
    return (int32_t*)(node_counter_[current_pipe_]);
  }

  int32_t* GetEdgeCounter() override {
    return (int32_t*)(edge_counter_[current_pipe_]);
  }

  int32_t* GetCacheSearchMap() override {
    return (int32_t*)(cache_search_map_[current_pipe_]);
  }

  vint* GetSparseMatrixSize() override {
    return (vint*)(sparseMatrix_size_[current_pipe_]);
  }

  void IpcCloseMemHandle() override {
    for (int32_t i = 0; i < 10; i++) {
      cudaIpcCloseMemHandle(bcsr_[current_pipe_][i]);
    }

    for (int32_t i = 0; i < 10; i++) {
      cudaIpcCloseMemHandle(bcsc_[current_pipe_][i]);
    }
    cudaCheckError();
  }

  vint** GetBCSRData() override {
    // 注意为了实现方便， sparseMatrixData 在存储数据时的顺序为 f_bcsr, f_bcsc, s_bcsr, s_bcsc，这里为了取数据在含义上解耦，把 bcsr 和 bcsc 分开了，所以取数据时的基址要确定好
    // bcsr_ 字段 IpcHandle
    for (int32_t i = 0; i < 5; i++) {
      cudaIpcOpenMemHandle(&bcsr_[current_pipe_][i], *(cudaIpcMemHandle_t*)&shm->sparseMatrixData[central_device][current_pipe_][i], cudaIpcMemLazyEnablePeerAccess);
      cudaIpcOpenMemHandle(&bcsr_[current_pipe_][5 + i], *(cudaIpcMemHandle_t*)&shm->sparseMatrixData[central_device][current_pipe_][10 + i], cudaIpcMemLazyEnablePeerAccess);
    }
    cudaCheckError();

    return (vint**)(bcsr_[current_pipe_].data());
  }

  vint** GetBCSCData() override {
    // bcsc_ 字段 IpcHandle
    for (int32_t i = 0; i < 5; i++) {
      cudaIpcOpenMemHandle(&bcsc_[current_pipe_][i], *(cudaIpcMemHandle_t*)&shm->sparseMatrixData[central_device][current_pipe_][5 + i], cudaIpcMemLazyEnablePeerAccess);
      cudaIpcOpenMemHandle(&bcsc_[current_pipe_][5 + i], *(cudaIpcMemHandle_t*)&shm->sparseMatrixData[central_device][current_pipe_][15 + i], cudaIpcMemLazyEnablePeerAccess);
    }
    cudaCheckError();

    return (vint**)(bcsc_[current_pipe_].data());
  }

  int64_t* GetFeatureAttribute() override {
    return featureAttribute_.data();
  }

  float* GetCpuFeatureCache() override {
    return (float*) (cpu_float_features_);
  }

  float** GetGpuFeatureCache() override {
    return (float**) (all_gpu_float_features_);
  }

  // debug
  void* GetHostCpuFeatureCache() override {
    return shared_data_;
  }
  // debug

  int32_t GetTrainStep() override {
    return train_step_;
  }
  int32_t GetValidStep() override {
    return valid_step_;
  }
  int32_t GetTestStep() override {
    return test_step_;
  }

  void Finalize() override {
    for(int i = 0; i < INTERBATCH_CON; i++){
      cudaIpcCloseMemHandle(ids_[i]);
      cudaIpcCloseMemHandle(float_features_[i]);
      cudaIpcCloseMemHandle(labels_[i]);
      cudaIpcCloseMemHandle(agg_src_[i]);
      cudaIpcCloseMemHandle(agg_dst_[i]);
      cudaIpcCloseMemHandle(node_counter_[i]);
      cudaIpcCloseMemHandle(edge_counter_[i]);
      sem_t* sem = semw_[i];
      if(sem_close(sem) == -1){
        std::cout<<"close sem "<<i<<" failed\n";
      }
    }

    cudaSetDevice(central_device);
    cudaFree(all_gpu_float_features_);
  }
private:
  std::vector<void*> ids_;
  std::vector<void*> float_features_;
  std::vector<void*> labels_;
  std::vector<void*> agg_src_;
  std::vector<void*> agg_dst_;
  std::vector<void*> node_counter_;
  std::vector<void*> edge_counter_;
  std::vector<sem_t*> semw_;
  std::vector<sem_t*> semr_;

  std::vector<void*> cache_search_map_;
  std::vector<void*> sparseMatrix_size_;
  std::vector<std::vector<void*>> bcsr_;
  std::vector<std::vector<void*>> bcsc_;
  std::vector<int64_t> featureAttribute_; // 0:cpu_float_feature_len; 1:gpu_node_capacity; 2:feature_dim
  void* cpu_float_features_;
  // debug
  void* shared_data_;                     // debug
  // debug
  void** all_gpu_float_features_;         // 当前 gpu 拥有的二级指针空间
  std::vector<void*> gpu_float_features_; // 当前 gpu 所能读取的本卡和其他卡的一级指针空间
  volatile shmStruct *shm;
  int central_device;
  bool is_opened_handle_bcsr_[INTERBATCH_CON];        // 标记是否存在有效的 IpcHandle，进而决定是否需要关闭已用 handle
  bool is_opened_handle_bcsc_[INTERBATCH_CON];        // 标记是否存在有效的 IpcHandle，进而决定是否需要关闭已用 handle

  int32_t train_step_;
  int32_t valid_step_;
  int32_t test_step_;
  int current_pipe_;

  inline void initialize_dataStructure_feature(int gpu_number) {
    // 需要确保 all_gpu_float_features_ 是在指定 device 上分配的空间
    cudaSetDevice(central_device);

    for (int i = 0; i < INTERBATCH_CON; i++) {
      is_opened_handle_bcsr_[i] = false;
      is_opened_handle_bcsc_[i] = false;
    }
    
    // 混合数据结构字段
    sparseMatrix_size_.resize(INTERBATCH_CON);
    bcsr_.resize(INTERBATCH_CON);
    bcsc_.resize(INTERBATCH_CON);
    for (int i = 0; i < INTERBATCH_CON; i++) {
      bcsr_[i].resize(10);
      bcsc_[i].resize(10);
    }

    // feature 获取相关字段
    cache_search_map_.resize(INTERBATCH_CON);
    featureAttribute_.resize(3);
    gpu_float_features_.resize(gpu_number);

    // 共享文件读取
    for(int i = 0; i < INTERBATCH_CON; i++) {
      sparseMatrix_size_[i] = (vint*)(shm->sparseMatrixMetric[central_device][i]);
      cudaIpcOpenMemHandle(&cache_search_map_[i], *(cudaIpcMemHandle_t*)&shm->memHandle[central_device][i][7], cudaIpcMemLazyEnablePeerAccess);
    }

    // cpu feature
    for (int i = 0; i < 3; i++) {
      featureAttribute_[i] = shm->featureAttribute[i];
    }

    const char* shm_name = "/cpu_features_shared";
    const int64_t SIZE = featureAttribute_[0] * sizeof(float);
    int shm_fd = shm_open(shm_name, O_RDWR, 0666);
    float* shared_data = (float*)mmap(0, SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);

    // debug
    shared_data_ = shared_data;
    // debug
    cudaHostRegister(shared_data, SIZE, cudaHostRegisterDefault);
    cudaHostGetDevicePointer((void**)&cpu_float_features_, (void*)shared_data, 0);
    
    // gpu feature 的二级指针 reconstruct
    cudaMalloc(&all_gpu_float_features_, gpu_number * sizeof(float*));
    for (int dev_id = 0; dev_id < gpu_number; dev_id++) {
      cudaIpcOpenMemHandle((void**) &gpu_float_features_[dev_id], *(cudaIpcMemHandle_t*)&shm->gpu_feature_memHandle[dev_id], cudaIpcMemLazyEnablePeerAccess); // 不区分 pipe，数据始终相同
    }
    cudaMemcpy(all_gpu_float_features_, gpu_float_features_.data(), sizeof(float*) * gpu_number, cudaMemcpyHostToDevice);
    cudaCheckError();
  }
};
IPCEnv* NewIPCEnv(){
  return new GPUIPCEnv();
}

// Define the GPU implementation that launches the CUDA kernel.

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
    ){
    int current_dev = -1;
    cudaGetDevice(&current_dev);
    auto device = "cuda:" + std::to_string(current_dev);
    cudaCheckError();

    cudaMemcpy(h_node_counter, node_counter, 16 * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_edge_counter, edge_counter, 16 * sizeof(int32_t), cudaMemcpyDeviceToHost);
    int hop_num = h_node_counter[INTRABATCH_CON * 3 - 1];

    std::vector<torch::Tensor> ret;

    torch::Tensor ids_tensor = torch::from_blob(
      ids,
      {(long long)h_node_counter[INTRABATCH_CON * 3 + hop_num]},
      torch::TensorOptions().dtype(torch::kI32).device(device));

    ret.push_back(ids_tensor);

    torch::Tensor feature_tensor = torch::from_blob(
      float_features,
      {(long long)(h_node_counter[INTRABATCH_CON * 3 + hop_num]), (long long)(feature_dim)},
      torch::TensorOptions().dtype(torch::kF32).device(device));

    ret.push_back(feature_tensor);

    torch::Tensor labels_tensor = torch::from_blob(
      labels,
      {(long long)h_node_counter[INTRABATCH_CON * 3]},
      torch::TensorOptions().dtype(torch::kI32).device(device));   

    ret.push_back(labels_tensor);

    for(int i = hop_num; i > 0; i--){
      torch::Tensor agg_src_tensor = torch::from_blob(
        agg_src,
          {(long long)h_edge_counter[INTRABATCH_CON * 3 + i]},
          torch::TensorOptions().dtype(torch::kI32).device(device));
      torch::Tensor agg_dst_tensor = torch::from_blob(
        agg_dst,
          {(long long)h_edge_counter[INTRABATCH_CON * 3 + i]},
          torch::TensorOptions().dtype(torch::kI32).device(device));
      ret.push_back(agg_src_tensor);
      ret.push_back(agg_dst_tensor);
    }

    return ret;
}
