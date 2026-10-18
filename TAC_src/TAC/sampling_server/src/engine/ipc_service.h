#ifndef IPC_ENV_H
#define IPC_ENV_H
#include <semaphore.h>
#include "buildinfo.h"

#include "acc-spmm.h"

class IPCEnv {
public:
  virtual void Coordinate(BuildInfo* info) = 0;
  virtual int32_t GetMaxStep() = 0;

  virtual void InitializeSamplesBuffer(int32_t batch_size, int32_t num_ids, int32_t feature_dim, int32_t device_id, int32_t pipeline_depth) = 0;
  virtual void InitializeFeaturesBuffer(int32_t batch_size, int32_t num_ids, int32_t feature_dim, int32_t device_id, int32_t pipeline_depth) = 0;
  virtual void FillUpBCSRBuffer(vint num_rows, vint num_cols, vint nnz, vint num_TCBlocks, vint num_rowWindows, int32_t device_id, int32_t current_pipe) = 0;
  virtual void FillUpBCSCBuffer(vint num_rows, vint num_cols, vint nnz, vint num_TCBlocks, vint num_colWindows, int32_t device_id, int32_t current_pipe) = 0;
  virtual void InitializeFeatureBuffer(float* cpu_float_features, int64_t cpu_float_feature_len, std::vector<float*> &all_float_feature_cache_ptr, int32_t gpu_node_capacity, int32_t feature_dim) = 0;

  virtual int32_t GetRawBatchsize() = 0;
  virtual int32_t GetLocalBatchId(int32_t global_batch_id) = 0;
  virtual int32_t GetCurrentBatchsize(int32_t dev_id, int32_t current_mode) = 0;
  virtual int32_t GetCurrentMode(int32_t global_batch_id) = 0;
  
  virtual int32_t* GetIds(int32_t dev_id, int32_t current_pipe) = 0;
  virtual float* GetFloatFeatures(int32_t dev_id, int32_t current_pipe) = 0;
  virtual int32_t* GetLabels(int32_t dev_id, int32_t current_pipe) = 0;
  virtual int32_t* GetAggSrc(int32_t dev_id, int32_t current_pipe) = 0;
  virtual int32_t* GetAggDst(int32_t dev_id, int32_t current_pipe) = 0;
  virtual int32_t* GetNodeCounter(int32_t dev_id, int32_t current_pipe) = 0;
  virtual int32_t* GetEdgeCounter(int32_t dev_id, int32_t current_pipe) = 0;
  virtual int32_t* GetFullCacheSearchBuffer(int32_t dev_id, int32_t current_pipe) = 0;
  virtual vint* GetBCSRRowWindowOffset(int32_t dev_id, int32_t current_pipe) = 0;
  virtual vint* GetBCSRTcOffset(int32_t dev_id, int32_t current_pipe) = 0;
  virtual vint* GetBCSRSparseA2B(int32_t dev_id, int32_t current_pipe) = 0;
  virtual TCLOCAL_TYPE* GetBCSRTcLocalBit(int32_t dev_id, int32_t current_pipe) = 0;
  virtual vint* GetBCSCColWindowOffset(int32_t dev_id, int32_t current_pipe) = 0;
  virtual vint* GetBCSCTcOffset(int32_t dev_id, int32_t current_pipe) = 0;
  virtual vint* GetBCSCSparseA2C(int32_t dev_id, int32_t current_pipe) = 0;
  virtual TCLOCAL_TYPE* GetBCSCTcLocalBit(int32_t dev_id, int32_t current_pipe) = 0;

  virtual void IPCPost(int32_t dev_id, int32_t current_pipe) = 0;
  virtual void IPCWait(int32_t dev_id, int32_t current_pipe) = 0;

  virtual void Finalize() = 0;
  virtual int32_t GetTrainStep() = 0;

};
 
IPCEnv* NewIPCEnv(int32_t device_count);

#endif