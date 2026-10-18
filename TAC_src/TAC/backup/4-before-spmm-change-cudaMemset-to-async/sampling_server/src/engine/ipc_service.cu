#include "ipc_service.h"

#include <iostream>
#include <cstdint>
#include <fstream>
#include <string>

#include <chrono>
#include <vector>
#include <semaphore.h>
#include <stdio.h>
#include <stdlib.h>

#include "helper_multiprocess.h"
#include "system_config.cuh"

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

typedef struct shmStruct_st {
  int32_t steps[3];
  vint sparseMatrixMetric[MAX_DEVICE][INTERBATCH_CON][10]; // 每个 device 在不同 batch 采样到的数据都是不同的
  int64_t featureAttribute[3];
  cudaIpcMemHandle_t memHandle[MAX_DEVICE][INTERBATCH_CON][MEMORY_USAGE];
  cudaIpcMemHandle_t sparseMatrixData[MAX_DEVICE][INTERBATCH_CON][8];
  cudaIpcMemHandle_t gpu_feature_memHandle[MAX_DEVICE];
} shmStruct;

class CUDAIPCEnv : public IPCEnv {
public:
  CUDAIPCEnv(int32_t device_count){
    // std::cout<<"start initialize ipc env\n";
    const char shmName[] = "fuse_simpleIPCshm";
    if (sharedMemoryCreate(shmName, sizeof(*shm_), &info_) != 0) {
      printf("Failed to create shared memory slab\n");
      exit(EXIT_FAILURE);
    }
    // std::cout<<"Shared Memory Opened\n";

    shm_ = (volatile shmStruct *)info_.addr;
    memset((void *)shm_, 0, sizeof(*shm_));
    
    ids_.resize(device_count);
    float_features_.resize(device_count);
    labels_.resize(device_count);
    agg_src_.resize(device_count);
    agg_dst_.resize(device_count);
    node_counter_.resize(device_count);
    edge_counter_.resize(device_count);
    device_count_ = device_count;

    semr_.resize(device_count);
    semw_.resize(device_count);

    // 新增数据指针数组
    bcsr_.resize(device_count);
    bcsc_.resize(device_count);
    full_cache_search_buffer_.resize(device_count);
  }
  
  void Coordinate(BuildInfo* info) override {
    // std::cout<<"Start Coordinate Data Parallel Params\n";
    int32_t partition_count = info->partition_count;
    epoch_ = info->epoch;
    raw_batch_size_ = info->raw_batch_size;

    // int32_t max_train_size = 0;
    // for(int32_t i = 0; i < partition_count; i++){
    //   int32_t ts_num = info->training_set_num[i];
    //   if(ts_num > max_train_size){
    //     max_train_size = ts_num;
    //   }
    // }
    int32_t min_train_size = 1000000000;
    for(int32_t i = 0; i < partition_count; i++){
      int32_t ts_num = info->training_set_num[i];
      if(ts_num < min_train_size){
        min_train_size = ts_num;
      }
    }


    train_step_ = (min_train_size - 1) / raw_batch_size_;

    // std::cout<<info->training_set_num[0]<<" "<<min_train_size<<" "<<train_step_<<" "<<raw_batch_size_<<"\n";

    for(int32_t i = 0; i < partition_count; i++){
      // train_batch_size_.push_back(((info->training_set_num[i] - 1)/train_step_ + 1));
      train_batch_size_.push_back(raw_batch_size_);
    }

    int32_t max_valid_size = 0;
    int32_t raw_valid_batch_size = 512;
    for(int32_t i = 0; i < partition_count; i++){
      int32_t ts_num = info->validation_set_num[i];
      if(ts_num > max_valid_size){
        max_valid_size = ts_num;
      }
    }
    // 上取整
    valid_step_ = (max_valid_size - 1) / raw_valid_batch_size + 1;
    for(int32_t i = 0; i < partition_count; i++){
      valid_batch_size_.push_back(((info->validation_set_num[i] - 1)/valid_step_ + 1));
      std::printf("[Info]: device-%d, valid_batch_size: %d\n", i, (info->validation_set_num[i] - 1)/valid_step_ + 1);
    }

    int32_t max_test_size = 0;
    int32_t raw_test_batch_size = 512;
    for(int32_t i = 0; i < partition_count; i++){
      int32_t ts_num = info->testing_set_num[i];
      if(ts_num > max_test_size){
        max_test_size = ts_num;
      }
    }
    test_step_ = (max_test_size - 1) / raw_test_batch_size + 1;
    for(int32_t i = 0; i < partition_count; i++){
      test_batch_size_.push_back(((info->testing_set_num[i] - 1)/test_step_ + 1));
      std::printf("[Info]: device-%d, test_batch_size: %d\n", i, (info->testing_set_num[i] - 1)/test_step_ + 1);
    }

    std::cout<<"Train Steps: "<<train_step_<<"\n";
    std::cout<<"Valid Steps: "<<valid_step_<<"\n";
    std::cout<<"Test Steps: "<<test_step_<<"\n";
    // for(int i = 0; i < partition_count; i++){
    //   std::cout<<"Train Bs "<<train_batch_size_[i]<<" Valid Bs "<<valid_batch_size_[i]<<" Test Bs "<<test_batch_size_[i]<<"\n"; 
    // }

    shm_->steps[0] = train_step_;
    shm_->steps[1] = valid_step_;
    shm_->steps[2] = test_step_;

  }

  int32_t GetMaxStep() override {
    return (((train_step_ + valid_step_) * epoch_) + test_step_);
  }

  void InitializeSamplesBuffer(int32_t batch_size, int32_t num_ids, int32_t feature_dim, int32_t device_id, int32_t pipeline_depth) override {
    cudaSetDevice(device_id);
    std::string ssr = "fuse_sem_r_";
    std::string ssw = "fuse_sem_w_";
    (semr_[device_id]).resize(pipeline_depth);
    (semw_[device_id]).resize(pipeline_depth);
    for(int32_t i = 0; i < pipeline_depth; i++){
      void* new_ids;
      cudaMalloc(&new_ids, num_ids * sizeof(int32_t));
      cudaCheckError();
      // void* new_features;
      // cudaMalloc(&new_features, int64_t(int64_t(int64_t(num_ids) * feature_dim) * sizeof(float)));
      // cudaCheckError();
      void* new_labels;
      cudaMalloc(&new_labels, batch_size * sizeof(int32_t));
      cudaCheckError();
      void* new_agg_src;
      cudaMalloc(&new_agg_src, num_ids * sizeof(int32_t));
      cudaCheckError();
      void* new_agg_dst;
      cudaMalloc(&new_agg_dst, num_ids * sizeof(int32_t));
      cudaCheckError();
      void* new_node_counter;
      cudaMalloc(&new_node_counter, 16 * sizeof(int32_t));
      cudaCheckError();
      void* new_edge_counter;
      cudaMalloc(&new_edge_counter, 16 * sizeof(int32_t));
      cudaCheckError();

      cudaIpcGetMemHandle((cudaIpcMemHandle_t *)&shm_->memHandle[device_id][i][0], new_ids);
      // cudaIpcGetMemHandle((cudaIpcMemHandle_t *)&shm_->memHandle[device_id][i][1], new_features);
      cudaIpcGetMemHandle((cudaIpcMemHandle_t *)&shm_->memHandle[device_id][i][2], new_labels);
      cudaIpcGetMemHandle((cudaIpcMemHandle_t *)&shm_->memHandle[device_id][i][3], new_agg_src);
      cudaIpcGetMemHandle((cudaIpcMemHandle_t *)&shm_->memHandle[device_id][i][4], new_agg_dst);
      cudaIpcGetMemHandle((cudaIpcMemHandle_t *)&shm_->memHandle[device_id][i][5], new_node_counter);
      cudaIpcGetMemHandle((cudaIpcMemHandle_t *)&shm_->memHandle[device_id][i][6], new_edge_counter);
      cudaCheckError();

      ids_[device_id].push_back(new_ids);
      // float_features_[device_id].push_back(new_features);
      labels_[device_id].push_back(new_labels);
      agg_src_[device_id].push_back(new_agg_src);
      agg_dst_[device_id].push_back(new_agg_dst);
      node_counter_[device_id].push_back(new_node_counter);
      edge_counter_[device_id].push_back(new_edge_counter);

      // 新增完整的 cache_index 用于确定当前 batch 的所有采样节点是否缓存在 gpu 上
      void* new_full_cache_search_buffer;
      cudaMalloc(&new_full_cache_search_buffer, num_ids * sizeof(int32_t));
      cudaIpcGetMemHandle((cudaIpcMemHandle_t *)&shm_->memHandle[device_id][i][7], new_full_cache_search_buffer);
      cudaCheckError();
      full_cache_search_buffer_[device_id].push_back(new_full_cache_search_buffer);

      // BCSR structure data
      int32_t num_rowWindows = 800000; // batch_size: 2e4, fanout: [100, 100], node: 50w, edge: 600w
      int32_t num_TCBlocks_bcsr = 100000;

      void* rowWindowOffset;
      cudaMalloc(&rowWindowOffset, (num_rowWindows + 1) * sizeof(vint)); // change2
      void* tcOffset_bcsr;
      cudaMalloc(&tcOffset_bcsr, (num_TCBlocks_bcsr + 1) * sizeof(vint));
      void* sparseA2B;
      cudaMalloc(&sparseA2B, (num_TCBlocks_bcsr * COL_WINDOW) * sizeof(vint));
      void* tcLocalBit_bcsr;
      cudaMalloc(&tcLocalBit_bcsr, num_TCBlocks_bcsr * sizeof(TCLOCAL_TYPE));
      cudaCheckError();

      cudaIpcGetMemHandle((cudaIpcMemHandle_t *)&shm_->sparseMatrixData[device_id][i][0], rowWindowOffset);
      cudaIpcGetMemHandle((cudaIpcMemHandle_t *)&shm_->sparseMatrixData[device_id][i][1], tcOffset_bcsr);
      cudaIpcGetMemHandle((cudaIpcMemHandle_t *)&shm_->sparseMatrixData[device_id][i][2], sparseA2B);
      cudaIpcGetMemHandle((cudaIpcMemHandle_t *)&shm_->sparseMatrixData[device_id][i][3], tcLocalBit_bcsr);
      cudaCheckError();
  
      bcsr_[device_id].push_back(rowWindowOffset);
      bcsr_[device_id].push_back(tcOffset_bcsr);
      bcsr_[device_id].push_back(sparseA2B);
      bcsr_[device_id].push_back(tcLocalBit_bcsr);

      // BCSC structure data
      int32_t num_colWindows = 800000; // batch_size: 2e4, fanout: [100, 100], node: 50w, edge: 600w
      int32_t num_TCBlocks_bcsc = 100000;

      void* colWindowOffset;
      cudaMalloc(&colWindowOffset, (num_colWindows + 1) * sizeof(vint));
      void* tcOffset_bcsc;
      cudaMalloc(&tcOffset_bcsc, (num_TCBlocks_bcsc + 1) * sizeof(vint));
      void* sparseA2C;
      cudaMalloc(&sparseA2C, (num_TCBlocks_bcsc * ROW_WINDOW) * sizeof(vint));
      void* tcLocalBit_bcsc;
      cudaMalloc(&tcLocalBit_bcsc, num_TCBlocks_bcsc * sizeof(TCLOCAL_TYPE));
      cudaCheckError();

      cudaIpcGetMemHandle((cudaIpcMemHandle_t *)&shm_->sparseMatrixData[device_id][i][4], colWindowOffset);
      cudaIpcGetMemHandle((cudaIpcMemHandle_t *)&shm_->sparseMatrixData[device_id][i][5], tcOffset_bcsc);
      cudaIpcGetMemHandle((cudaIpcMemHandle_t *)&shm_->sparseMatrixData[device_id][i][6], sparseA2C);
      cudaIpcGetMemHandle((cudaIpcMemHandle_t *)&shm_->sparseMatrixData[device_id][i][7], tcLocalBit_bcsc);
      cudaCheckError();
  
      bcsc_[device_id].push_back(colWindowOffset);
      bcsc_[device_id].push_back(tcOffset_bcsc);
      bcsc_[device_id].push_back(sparseA2C);
      bcsc_[device_id].push_back(tcLocalBit_bcsc);

      //memory lock
      std::string ssri = ssr + std::to_string(device_id) + "_" + std::to_string(i);
      std::string sswi = ssw + std::to_string(device_id) + "_" + std::to_string(i);
      semr_[device_id][i] = sem_open(ssri.c_str(), O_CREAT | O_RDWR, 0666, 0);
      if (semr_[device_id][i] == SEM_FAILED ){
        printf("[Error] sem errno = %d\n", errno );
        return;
      }
      semw_[device_id][i] = sem_open(sswi.c_str(), O_CREAT | O_RDWR, 0666, 0);
      if (semw_[device_id][i] == SEM_FAILED){
        printf("[Error] sem errno = %d\n", errno );
        return;
      }
    }
    pipeline_depth_ = pipeline_depth;
  }

  void InitializeFeaturesBuffer(int32_t batch_size, int32_t num_ids, int32_t feature_dim, int32_t device_id, int32_t pipeline_depth) override {
    cudaSetDevice(device_id);
    for(int32_t i = 0; i < pipeline_depth; i++){
      void* new_features;
      cudaMalloc(&new_features, int64_t(int64_t(int64_t(num_ids) * feature_dim) * sizeof(float)));
      cudaCheckError();
      cudaIpcGetMemHandle((cudaIpcMemHandle_t *)&shm_->memHandle[device_id][i][1], new_features);
      float_features_[device_id].push_back(new_features);
    }
  }

  void FillUpBCSRBuffer(vint num_rows, vint num_cols, vint nnz, vint num_TCBlocks, vint num_rowWindows, int32_t device_id, int32_t current_pipe) override {
    shm_->sparseMatrixMetric[device_id][current_pipe][0] = num_rows;
    shm_->sparseMatrixMetric[device_id][current_pipe][1] = num_cols;
    shm_->sparseMatrixMetric[device_id][current_pipe][2] = nnz;
    shm_->sparseMatrixMetric[device_id][current_pipe][3] = num_TCBlocks;
    shm_->sparseMatrixMetric[device_id][current_pipe][4] = num_rowWindows;
  }

  void FillUpBCSCBuffer(vint num_rows, vint num_cols, vint nnz, vint num_TCBlocks, vint num_colWindows, int32_t device_id, int32_t current_pipe) override {
    shm_->sparseMatrixMetric[device_id][current_pipe][5] = num_rows;
    shm_->sparseMatrixMetric[device_id][current_pipe][6] = num_cols;
    shm_->sparseMatrixMetric[device_id][current_pipe][7] = nnz;
    shm_->sparseMatrixMetric[device_id][current_pipe][8] = num_TCBlocks;
    shm_->sparseMatrixMetric[device_id][current_pipe][9] = num_colWindows;
  }

  void InitializeFeatureBuffer(float* cpu_float_features, int64_t cpu_float_feature_len, std::vector<float*> &all_float_feature_cache_ptr, int32_t gpu_node_capacity, int32_t feature_dim) {
    // 参数
    shm_->featureAttribute[0] = cpu_float_feature_len;
    shm_->featureAttribute[1] = gpu_node_capacity;
    shm_->featureAttribute[2] = feature_dim;

    // cpu_feature_buffer（无法直接通过 Ipc Handle，需要通过 POSIX 共享内存）
    const char* shm_name = "/cpu_features_shared";
    const int64_t SIZE = cpu_float_feature_len * sizeof(float);
    // 创建共享内存
    int shm_fd = shm_open(shm_name, O_CREAT | O_RDWR, 0666);
    ftruncate(shm_fd, SIZE);
    float* shared_data = (float*)mmap(0, SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    // 填充数据
    std::memcpy(shared_data, cpu_float_features, SIZE);

    // gpu_feature_buffer
    for (int i = 0; i < device_count_; i++) {
      cudaIpcGetMemHandle((cudaIpcMemHandle_t *)&shm_->gpu_feature_memHandle[i], (void*) all_float_feature_cache_ptr[i]);
      cudaCheckError();
    }
  }

  int32_t GetRawBatchsize() override {
    return raw_batch_size_;
  }

  int32_t GetLocalBatchId(int32_t global_batch_id) override {
    int32_t local_batch_id = -1;
    if((global_batch_id) < ((train_step_ + valid_step_) * epoch_)){//train & valid
      int32_t epoch_batch_id = global_batch_id % (train_step_ + valid_step_);
      if(epoch_batch_id < train_step_){
        local_batch_id = epoch_batch_id;
      }else{
        local_batch_id = epoch_batch_id - train_step_;
      }
    }else{//test
      int32_t epoch_batch_id = (global_batch_id - ((train_step_ + valid_step_) * epoch_))%test_step_;
      local_batch_id = epoch_batch_id;
    }
    return local_batch_id;
  }

  int32_t GetCurrentBatchsize(int32_t dev_id, int32_t current_mode) override {
    if(current_mode == TRAINMODE){//train
      return train_batch_size_[dev_id];
    }else if(current_mode == VALIDMODE){//valid
      return valid_batch_size_[dev_id];
    }else{//test
      return test_batch_size_[dev_id];
    }
  }

  //tape_id / shard_count = batch_id
  int32_t GetCurrentMode(int32_t global_batch_id) override {
    int32_t current_mode;
    if((global_batch_id) < ((train_step_ + valid_step_) * epoch_)){//train & valid
      int32_t epoch_batch_id = global_batch_id % (train_step_ + valid_step_);
      if(epoch_batch_id < train_step_){
        current_mode = TRAINMODE;
      }else{
        current_mode = VALIDMODE;
      }
    }else{//test
      current_mode = TESTMODE;
    }
    return current_mode;
  }

  int32_t* GetIds(int32_t dev_id, int32_t current_pipe) override {
    return (int32_t*)(ids_[dev_id][current_pipe%pipeline_depth_]);
  }

  float* GetFloatFeatures(int32_t dev_id, int32_t current_pipe) override {
    return (float*)(float_features_[dev_id][current_pipe%pipeline_depth_]);
  }

  int32_t* GetLabels(int32_t dev_id, int32_t current_pipe) override {
    return (int32_t*)(labels_[dev_id][current_pipe%pipeline_depth_]);
  }

  int32_t* GetAggSrc(int32_t dev_id, int32_t current_pipe) override {
    return (int32_t*)(agg_src_[dev_id][current_pipe%pipeline_depth_]);
  }

  int32_t* GetAggDst(int32_t dev_id, int32_t current_pipe) override {
    return (int32_t*)(agg_dst_[dev_id][current_pipe%pipeline_depth_]);
  }

  int32_t* GetNodeCounter(int32_t dev_id, int32_t current_pipe) override {
    return (int32_t*)(node_counter_[dev_id][current_pipe%pipeline_depth_]);
  }

  int32_t* GetEdgeCounter(int32_t dev_id, int32_t current_pipe) override {
    return (int32_t*)(edge_counter_[dev_id][current_pipe%pipeline_depth_]);
  }

  int32_t* GetFullCacheSearchBuffer(int32_t dev_id, int32_t current_pipe) override {
    return (int32_t*)(full_cache_search_buffer_[dev_id][current_pipe%pipeline_depth_]);
  }

  // BCSR
  vint* GetBCSRRowWindowOffset(int32_t dev_id, int32_t current_pipe) override {
      return (vint*) bcsr_[dev_id][current_pipe * 4];
  }
  vint* GetBCSRTcOffset(int32_t dev_id, int32_t current_pipe) override {
      return (vint*) bcsr_[dev_id][current_pipe * 4 + 1];
  }
  vint* GetBCSRSparseA2B(int32_t dev_id, int32_t current_pipe) override {
      return (vint*) bcsr_[dev_id][current_pipe * 4 + 2];
  }
  TCLOCAL_TYPE* GetBCSRTcLocalBit(int32_t dev_id, int32_t current_pipe) override {
      return (TCLOCAL_TYPE*) bcsr_[dev_id][current_pipe * 4 + 3];
  }

  // BCSC
  vint* GetBCSCColWindowOffset(int32_t dev_id, int32_t current_pipe) override {
      return (vint*) bcsc_[dev_id][current_pipe * 4];
  }
  vint* GetBCSCTcOffset(int32_t dev_id, int32_t current_pipe) override {
      return (vint*) bcsc_[dev_id][current_pipe * 4 + 1];
  }
  vint* GetBCSCSparseA2C(int32_t dev_id, int32_t current_pipe) override {
      return (vint*) bcsc_[dev_id][current_pipe * 4 + 2];
  }
  TCLOCAL_TYPE* GetBCSCTcLocalBit(int32_t dev_id, int32_t current_pipe) override {
      return (TCLOCAL_TYPE*) bcsc_[dev_id][current_pipe * 4 + 3];
  }

  void IPCPost(int32_t dev_id, int32_t current_pipe) override {
    sem_t* sem = semw_[dev_id][current_pipe];
    sem_post(sem);
  }

  void IPCWait(int32_t dev_id, int32_t current_pipe) override {
    sem_t* sem = semr_[dev_id][current_pipe];
    sem_wait(sem);
  }

  void Finalize() override {
    for(int32_t i = 0; i < device_count_; i++){
      cudaSetDevice(i);
      for(int32_t j = 0; j < INTERBATCH_CON; j++){
        cudaFree(ids_[i][j]);
        cudaFree(float_features_[i][j]);
        cudaFree(labels_[i][j]);
        cudaFree(agg_src_[i][j]);
        cudaFree(agg_dst_[i][j]);
        cudaFree(node_counter_[i][j]);
        cudaFree(edge_counter_[i][j]);
        cudaCheckError();

        // 新增数据 cudaFree
        cudaFree(full_cache_search_buffer_[i][j]);

        sem_t* sem = semw_[i][j];
        if(sem_close(sem) == -1){
          std::cout<<"close sem "<<i<<" "<<j<<" failed\n";
        }
        std::string ssr = "fuse_sem_r_";
        std::string ssw = "fuse_sem_w_";
        std::string ssri = ssr + std::to_string(i) + "_" + std::to_string(j);
        std::string sswi = ssw + std::to_string(i) + "_" + std::to_string(j);
        sem_unlink(ssri.c_str());
        sem_unlink(sswi.c_str());
      }

      // 每个 device 在每个 pipe 有 4 个数据，两个 pipe 共 8 个
      for (int32_t j = 0; j < 8; j++) {
        cudaFree(bcsr_[i][j]);
        cudaFree(bcsc_[i][j]);
      }
    }

    sharedMemoryClose(&info_);
  }

  int32_t GetTrainStep() override {
    return train_step_;
  }
  
private:
  volatile shmStruct *shm_;
  sharedMemoryInfo info_;
  std::vector<std::vector<void*>> ids_;
  std::vector<std::vector<void*>> float_features_;
  std::vector<std::vector<void*>> labels_;
  std::vector<std::vector<void*>> agg_src_;
  std::vector<std::vector<void*>> agg_dst_;
  std::vector<std::vector<void*>> node_counter_;
  std::vector<std::vector<void*>> edge_counter_;
  std::vector<std::vector<sem_t*>> semr_;
  std::vector<std::vector<sem_t*>> semw_; 

  // 新增 IpcHandle 数据
  //std::vector<std::vector<void*>> bcsr_;
  //std::vector<std::vector<void*>> bcsc_;
  std::vector<std::vector<void*>> bcsr_;
  std::vector<std::vector<void*>> bcsc_;
  std::vector<std::vector<void*>> full_cache_search_buffer_;

  int32_t raw_batch_size_;
  std::vector<int32_t> train_batch_size_;
  std::vector<int32_t> valid_batch_size_;
  std::vector<int32_t> test_batch_size_;

  int32_t device_count_;

  int32_t train_step_;
  int32_t valid_step_;
  int32_t test_step_;

  int32_t epoch_;
  int32_t pipeline_depth_;
};

IPCEnv* NewIPCEnv(int32_t device_count){
  return new CUDAIPCEnv(device_count);
}
