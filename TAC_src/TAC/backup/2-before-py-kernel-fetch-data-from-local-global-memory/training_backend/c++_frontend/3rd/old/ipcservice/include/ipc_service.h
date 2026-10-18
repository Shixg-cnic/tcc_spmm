#include <vector>
#include <cstdint>
#include <semaphore.h>
#include "config.h"
#include <torch/extension.h>

#define INTRABATCH_CON 3
#define INTERBATCH_CON 2

class IPCEnv {
public:
  virtual int Initialize(int gpu_number) = 0;
  virtual int32_t* GetIds() = 0;
  virtual float* GetFloatFeatures() = 0;
  virtual int32_t* GetLabels() = 0;
  virtual int32_t* GetAggSrc() = 0;
  virtual int32_t* GetAggDst() = 0;
  virtual int32_t* GetNodeCounter() = 0;
  virtual int32_t* GetEdgeCounter() = 0;
  virtual int32_t* GetCacheSearchMap() = 0;
  virtual vint* GetSparseMatrixSize() = 0;
  virtual void IpcCloseMemHandle() = 0;
  virtual vint** GetBCSRData() = 0;
  virtual vint** GetBCSCData() = 0;
  virtual int64_t* GetFeatureAttribute() = 0;
  virtual float* GetCpuFeatureCache() = 0;
  virtual float** GetGpuFeatureCache() = 0;
  // debug
  virtual void* GetHostCpuFeatureCache() = 0;
  // debug
  virtual int32_t GetTrainStep() = 0;
  virtual int32_t GetValidStep() = 0;
  virtual int32_t GetTestStep() = 0;
  virtual void Finalize() = 0;
  virtual void Wait() = 0;
  virtual void Post() = 0;
};
IPCEnv* NewIPCEnv();

void InitializeIPC(int gpu_number);
std::vector<int32_t> get_steps();
std::vector<torch::Tensor> get_next(int feature_dim);
std::vector<int> get_block_size();
std::vector<vint> get_sparseMatrix_size();
std::vector<torch::Tensor> get_sparseMatrix_data();
std::vector<torch::Tensor> get_memoryaccess_data();
std::vector<int64_t> get_feature_attribute();
std::vector<void*> get_feature_cache_ptr();
void* get_host_cpu_feature_cache_ptr();
void Synchronize();