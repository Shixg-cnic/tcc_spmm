#include "operator_impl.cuh"
#include "device_launch_parameters.h"
#include "system_config.cuh"

#include <cuda_runtime.h>
#include <iostream>
#include <random>

#include <thrust/random/uniform_int_distribution.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/binary_search.h>

#include "ipc_service.h"

#include <fstream>
#include <chrono>

#include <cuda_profiler_api.h>

#define OP_THREAD_NUM 1024
#define SH_MEM_SIZE 1024

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

//assume no duplicate
__global__ void batch_generate(
	int32_t* batch_ids, 
	int32_t* labels, 
	int32_t batch_size, 
	int32_t counter, 
	int32_t* all_ids, 
	int32_t* all_labels, 
	int32_t total_cap,
	int32_t* position_map,
	uint32_t* accessed_map,
	int32_t dev_id,
	bool is_presc)
{
	int32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
	if(idx < batch_size){
		if((batch_size * counter + idx) >= total_cap){
			batch_ids[idx] = -1;
			labels[idx] = -1;
		}else{
			int32_t src_id = all_ids[(batch_size * counter + idx)%(total_cap)];
			batch_ids[idx] = src_id;
			// accessed_map[src_id] = 1;
			int32_t bitmap_idx = src_id / 32;
			int32_t bitmap_off = src_id % 32;
			uint32_t bitmap_data = (1 << bitmap_off);
			atomicOr(accessed_map + bitmap_idx, bitmap_data);
			position_map[src_id] = idx;
			labels[idx] = all_labels[(batch_size * counter + idx)%(total_cap)];

			//if (!is_presc && dev_id == 0 && idx == 0){
			//	printf(">>batch id:\n\t");
			//	for(int i=0; i < batch_size; i++){
			//	    printf("%d ", batch_ids[i]);
			//    }
			//	printf("\n");
			//}
		}	
	}
}

__global__ void counter_update(
	int32_t* node_counter, 			// for the inputs of feature extraction and graph sampling
	int32_t* edge_counter, 			// for the inputs of graph sampling
	int32_t op_id, 
	int32_t size,
	int32_t hop_num)
{
	if(op_id == 0){
		node_counter[0] = 0;										//start offset for feature extraction
		node_counter[1] = size;										//sampled nodes num for feature extraction
		node_counter[INTRABATCH_CON * 3 + (op_id / INTRABATCH_CON)] = node_counter[0] + node_counter[1];
		node_counter[INTRABATCH_CON * 3 - 1] = hop_num;
	}else if((op_id > 0) && (op_id % INTRABATCH_CON == 0)){
		node_counter[0] = node_counter[0] + node_counter[1];		
		node_counter[1] = node_counter[INTRABATCH_CON * 2];			
		node_counter[INTRABATCH_CON * 2] = 0;						//reset
		node_counter[INTRABATCH_CON * 2 + 1] = node_counter[0] + node_counter[1];					//total
		//hop1 start (0 C0), hop2 start (C0 C1), hop3 start (C0+C1 C2)
		//hop1 done (C0 C1), hop2 done (C0+C1 C2), hop3 done (C0+C1+C2 C3)
		edge_counter[0] = edge_counter[0] + edge_counter[1];		//input offset for graph sampling
		edge_counter[1] = edge_counter[2];							//input vertex size of graph sampling
		edge_counter[2] = 0;										//reset
		//hop1 start (0 0 0), hop2 start (0 B1 0), hop3 start (B1 B2 0)
		//hop1 done (0 B1 0), hop2 done (B1 B2 0), hop3 done (B1+B2 B3 0)
		node_counter[INTRABATCH_CON * 3 + (op_id / INTRABATCH_CON)] = node_counter[0] + node_counter[1];
		edge_counter[INTRABATCH_CON * 3 + (op_id / INTRABATCH_CON)] = edge_counter[0] + edge_counter[1];	
	}else if(op_id % INTRABATCH_CON > 0){
		node_counter[(op_id % INTRABATCH_CON) * 2] = node_counter[0];										
		node_counter[(op_id % INTRABATCH_CON) * 2 + 1] = node_counter[1];	
	}else{
		printf("Sampling Parameters Error\n");
	}
}

__global__ void fillFullCacheSearchBuffer(
	int32_t *cache_index,
	int32_t *full_cache_index,
	int32_t node_off,	
	int32_t batch_size,
	int32_t op_id,
	int32_t dev_id) 
{
	for (int32_t idx = threadIdx.x + blockDim.x * blockIdx.x; idx < batch_size; idx += gridDim.x * blockDim.x) {
		full_cache_index[node_off + idx] = cache_index[idx];
	}
}

extern "C"
void BatchGenerate(
	cudaStream_t    strm_hdl, 
	FeatureStorage* feature,
	UnifiedCache*   cache,
	MemoryPool*     memorypool,
	int32_t         batch_size, 
	int32_t         counter, 
	int32_t         part_id,
	int32_t         dev_id,
	int32_t         mode,
	bool 			is_presc,
	int32_t 		hop_num)
{
	if (!is_presc) {
		//printf("\t[Info] device-%d BatchGenerate\n", dev_id);
	}

	int32_t* all_ids 	= nullptr;
	int32_t* all_labels = nullptr;
	int32_t total_cap 	= 0;
	if(mode == TRAINMODE){
		all_ids 	= feature->GetTrainingSetIds(dev_id);
		all_labels 	= feature->GetTrainingLabels(dev_id);
		total_cap 	= feature->TrainingSetSize(dev_id);
	}else if(mode == VALIDMODE){
		all_ids 	= feature->GetValidationSetIds(dev_id);
		all_labels 	= feature->GetValidationLabels(dev_id);
		total_cap 	= feature->ValidationSetSize(dev_id);
	}else if(mode == TESTMODE){
		all_ids 	= feature->GetTestingSetIds(dev_id);
		all_labels 	= feature->GetTestingLabels(dev_id);
		total_cap 	= feature->TestingSetSize(dev_id);
	}else{
		std::cout<<"invalid mode: "<<mode<<"\n";
	}

	int32_t total_node_num 		= feature->TotalNodeNum();

	int32_t* batch_ids 			= memorypool->GetSampledIds();
	int32_t* labels 			= memorypool->GetLabels();
	uint32_t* accessed_map 		= memorypool->GetAccessedMap();
	int32_t* position_map 		= memorypool->GetPositionMap();
	int32_t* node_counter 		= memorypool->GetNodeCounter();
	int32_t* edge_counter 		= memorypool->GetEdgeCounter();
	int32_t* agg_src_ids 		= memorypool->GetAggSrcId();
	int32_t* agg_dst_ids 		= memorypool->GetAggDstId();
	int32_t* agg_src_off 		= memorypool->GetAggSrcOf();
	int32_t* agg_dst_off 		= memorypool->GetAggDstOf();
	int32_t* cache_index 		= memorypool->GetCacheSearchBuffer();
	int32_t* full_cache_index 	= memorypool->GetFullCacheSearchBuffer();
	int32_t* sampled_ids 		= memorypool->GetSampledIds();
	float* dst_float_buffer 	= memorypool->GetFloatFeatures();
	int32_t op_id				= 0;

	if(all_ids == nullptr){
		std::cout<<"invalid src id ptr\n";
		return;
	}
	if(all_labels == nullptr){
		std::cout<<"invalid label ptr\n";
		return;
	}
	cudaCheckError();

	cudaMemsetAsync(accessed_map, 0, int64_t(int64_t((total_node_num / 32) + 1) * int64_t(sizeof(uint32_t))), (strm_hdl));
	//cudaMemsetAsync(position_map, 0, int64_t(int64_t(total_node_num) * int64_t(sizeof(int32_t))), (strm_hdl));
	cudaCheckError();
	// cache->Finalize(dev_id);
	cudaMemsetAsync(node_counter, 0, 16 * sizeof(int32_t), (strm_hdl));
	cudaMemsetAsync(edge_counter, 0, 16 * sizeof(int32_t), (strm_hdl));
	cudaCheckError();

	// batch_size: 输入参数指定的批次大小；total_cap：总的数据量；总共 total_cap 的数据会逐一批次进行处理，需要计算当前批次的实际可用数据量
	int32_t size = ((batch_size*(counter+1)) >= total_cap) ? (total_cap - batch_size * counter) : batch_size;
	dim3 bg_block((size - 1)/OP_THREAD_NUM + 1, 1);
	dim3 bg_thread(OP_THREAD_NUM, 1);
	batch_generate<<<bg_block, bg_thread, 0, (strm_hdl)>>>(batch_ids, labels, size, counter, all_ids, all_labels, total_cap, position_map, accessed_map, dev_id, is_presc);
	cudaCheckError();

	counter_update<<<1, 1, 0, (strm_hdl)>>>(node_counter, edge_counter, 0, size, hop_num);
	cudaCheckError();
	if(!is_presc){
		cache->FindFeat(sampled_ids, cache_index, node_counter, op_id, strm_hdl, dev_id);

		// 当前 hop 的 cache_index 结果保存
		// 这里还是采用在 kernel 外获取 counter 数据，kernel 内访问确实不需要gpu->cpu的访存，但是不如这种简单
		int32_t* h_node_counter = (int32_t*)malloc(64);
		cudaMemcpy(h_node_counter, node_counter, 64, cudaMemcpyDeviceToHost);

		// batchGenerate 或者 当前 hop 采样到的节点在 sampled_ids 中的范围：[node_off, node_off + batch_size)
		int32_t node_off = h_node_counter[(op_id % INTRABATCH_CON) * 2];
		int32_t batch_size = h_node_counter[(op_id % INTRABATCH_CON) * 2 + 1];

		//===== debug
		//thrust::device_vector<int32_t> d_cache_index(cache_index, cache_index + batch_size);
		//thrust::host_vector<int32_t> h_cache_index = d_cache_index;
		//std::printf("batch_generate: cache_index[%d~%d]: { ", node_off, node_off + batch_size);
		//for (int i = 0; i < batch_size; i++) {
		//	std::printf("%d, ", h_cache_index[i]);
		//}
		//std::printf("}\n");
		//===== debug

		// 首先需要明确一点，sampled_ids 和 full_cache_search_buffer 是逐位对应的，所以在 FindFeat 中对于 sampled_ids 的访问范围也就是这一个操作生成或者采样得到的节点范围
		// 每个 thread 填充一个位置
		dim3 ffc_block(16, 1, 1); // 这里的 16 参考了 randomSample 的设置，因为 sampled_ids 和 cache_index 是一一对应的(况且 randomSample 中 thread 的视角还是一个 sampled_ids 中的一个节点的一个邻居， 访问 cache_index 的视角还是一个节点，所以它的配置够用，这里一定也够用)
		dim3 ffc_thread(OP_THREAD_NUM, 1);	
		fillFullCacheSearchBuffer<<<ffc_block, ffc_thread, 0, (strm_hdl)>>>(cache_index, full_cache_index, node_off, batch_size, op_id, dev_id);
		
		// FindFeat 在哈希表中查询 sampled_ids[node_off, node+batch_size) 的数据并将结果写入到 cache_index[0, batch_size)（但是不确定这个指定流是否可用，文档中并没有提到这个，只能在源码中查询到)
		//thrust::copy(thrust::cuda::par.on(strm_hdl), cache_index, cache_index + batch_size, full_cache_index + node_off);
		cudaCheckError();
	}

}

/////////random sampler//////////
__global__ void random_sample(
	int32_t*  sampled_ids,
	int32_t   op_id,
	int64_t** csr_node_index, 
	int32_t** csr_dst_node_ids,
	char*     partition_index,
	int32_t*  parition_offset,
	int32_t   count, 
	int32_t   partition_count, 
	int32_t*  agg_src_ids,
	int32_t*  agg_dst_ids,
	uint32_t*  accessed_map,
	int32_t*  position_map,
	int32_t*  node_counter,
	int32_t*  edge_counter,
	int32_t   dev_id
	)
{	
	/*the direction for agg is reversed*/
	__shared__ int32_t sh_agg_src_ids[SH_MEM_SIZE];
	__shared__ int32_t sh_agg_dst_ids[SH_MEM_SIZE];
	__shared__ int32_t sh_sampled_id[SH_MEM_SIZE];
	/*local offset: 0, local node offset; 1, global node offset; 2, local edge offset; 3, global edge offset */
	__shared__ int32_t local_offset[4];
	int32_t* input_ids = nullptr;
	int32_t batch_size = 0;
	if(op_id == INTRABATCH_CON){
		input_ids = sampled_ids;
		batch_size = node_counter[1];
	}else if(op_id > INTRABATCH_CON){
		input_ids = agg_src_ids + edge_counter[0];
		batch_size = edge_counter[1];
	}
	for(int32_t idx = threadIdx.x + blockDim.x * blockIdx.x; idx < batch_size * count; idx += gridDim.x * blockDim.x){
		if(threadIdx.x == 0){
            local_offset[0] = 0;
			local_offset[1] = 0;
            local_offset[2] = 0;
            local_offset[3] = 0;
        }
        __syncthreads();
		int32_t sample_src_id = input_ids[idx/count];
		int32_t sample_dst_id;
		if(sample_src_id >= 0){
			int32_t neighbor_offset = idx%count;
			int32_t part_id = partition_index[idx/count];
			int32_t part_offset = parition_offset[idx/count];
			int64_t start_index;
			int32_t col_size;
			if(part_id < 0){
				start_index = csr_node_index[partition_count][sample_src_id];
				col_size = csr_node_index[partition_count][(sample_src_id + 1)] - start_index;
			}else{
				start_index = csr_node_index[part_id][part_offset];
				col_size = csr_node_index[part_id][(part_offset + 1)] - start_index;
			}

			if(neighbor_offset >= col_size){
				sample_dst_id = -1;
			}else{
				//if (dev_id == 0) {
				//	printf("sample_src_id: %d, part_id: %d, col_size: %d\n", sample_src_id, part_id, col_size);
				//}
				thrust::minstd_rand engine;
				//thrust::minstd_rand engine(12345); // debug 固定随机数
				engine.discard(idx);
				//engine.discard(10); // debug 固定随机数
				thrust::uniform_int_distribution<> dist(0, col_size - 1);
				int32_t dst_index = dist(engine);
				if(part_id < 0){
					sample_dst_id = csr_dst_node_ids[partition_count][(int64_t(start_index + int64_t(dst_index)))];
				}else{
					sample_dst_id = csr_dst_node_ids[part_id][(int64_t(start_index + int64_t(dst_index)))];
				}
				if(sample_dst_id >= 0){
					int32_t bitmap_idx = sample_dst_id / 32;
					int32_t bitmap_off = sample_dst_id % 32;
					uint32_t bitmap_data = (1 << bitmap_off);
					uint32_t old_bitmap_data = atomicOr(accessed_map + bitmap_idx, bitmap_data);
					uint32_t is_accessed = (old_bitmap_data >> bitmap_off) % 2;
					// int32_t acc_count = atomicAdd(accessed_map + sample_dst_id, 1);
					if(is_accessed == 0){//first time node is sampled
						int32_t node_off = atomicAdd(local_offset, 1);
						sh_sampled_id[node_off] = sample_dst_id;

						// 重复边不采集（我们期望的边去重是去除一个顶点所采集到的重复节点对应的边，但是目前的点去重是保证 sampled_ids 中的节点唯一，所以这会导致对应到相同目标节点与源节点的边只会采样到一次）
						//int32_t edge_off = atomicAdd(local_offset + 2, 1);
						//sh_agg_src_ids[edge_off] = sample_dst_id;
						//sh_agg_dst_ids[edge_off] = sample_src_id;
					}
					int32_t edge_off = atomicAdd(local_offset + 2, 1);
					// 目前的方案是：对原始图的边进行翻转；legion 仍然采出边，不过由于原始数据中边进行了翻转，因此采集到的出边实际是入边；根据 src 和 dst 构建的拓扑矩阵保存的本质是入边信息，在矩阵乘进行信息聚合时就能够将信息聚集到 src
					sh_agg_src_ids[edge_off] = sample_dst_id;
					sh_agg_dst_ids[edge_off] = sample_src_id;
					//sh_agg_src_ids[edge_off] = sample_src_id;
					//sh_agg_dst_ids[edge_off] = sample_dst_id;
				}
			}
		}
		__syncthreads();
		if(threadIdx.x == 0){
            local_offset[1] = atomicAdd(node_counter + INTRABATCH_CON * 2, local_offset[0]);//global node count current hop
			local_offset[3] = atomicAdd(edge_counter + 2, local_offset[2]);//global edge count current hop
		}
        __syncthreads();
		if(threadIdx.x < local_offset[0]){
			int32_t node_base = local_offset[1] + node_counter[0] + node_counter[1];
			int32_t dst_id = sh_sampled_id[threadIdx.x]; 
			sampled_ids[node_base + threadIdx.x] = dst_id;
			position_map[dst_id] = node_base + threadIdx.x;
		}
		__syncthreads();
		if(threadIdx.x < local_offset[2]){
			int32_t edge_base = local_offset[3] + edge_counter[0] + edge_counter[1];
			agg_src_ids[edge_base + threadIdx.x] = sh_agg_src_ids[threadIdx.x];
			agg_dst_ids[edge_base + threadIdx.x] = sh_agg_dst_ids[threadIdx.x]; 
		}
		__syncthreads();
	}
}

__global__ void construct_graph(int32_t* agg_src_ids, int32_t* agg_dst_ids, 
								int32_t* agg_src_off, int32_t* agg_dst_off,
								int32_t* position_map, int32_t* edge_counter, int32_t* node_counter, int32_t op_id, int32_t dev_id, bool is_presc){
	int32_t edge_num = edge_counter[2];
	int32_t edge_off = edge_counter[0] + edge_counter[1];
	for(int32_t idx = threadIdx.x + blockDim.x * blockIdx.x; idx < edge_num; idx += gridDim.x * blockDim.x){
		int32_t src_id = agg_src_ids[edge_off + idx];
		int32_t dst_id = agg_dst_ids[edge_off + idx];
		int32_t src_of = position_map[src_id];
		int32_t dst_of = position_map[dst_id];
		agg_src_off[edge_off + idx] = src_of;
		agg_dst_off[edge_off + idx] = dst_of;

		//if (!is_presc && dev_id == 0){
        //    printf("src: %d --> %d, dst: %d ---> %d\n", src_of, src_id, dst_of, dst_id);
		//}
	}
}



/////////random sampler//////////
__global__ void pre_sample(
	int32_t*  sampled_ids,
	int32_t   op_id,
	int64_t* csr_node_index, 
	int32_t* csr_dst_node_ids,
	char*     partition_index,
	int32_t*  parition_offset,
	int32_t   count, 
	int32_t   partition_count, 
	int32_t*  agg_src_ids,
	int32_t*  agg_dst_ids,
	uint32_t*  accessed_map,
	int32_t*  position_map,
	int32_t*  node_counter,
	int32_t*  edge_counter,
	int32_t   dev_id,
	unsigned long long int*  edge_access_time
	)
{	
	/*the direction for agg is reversed*/
	__shared__ int32_t sh_agg_src_ids[SH_MEM_SIZE];
	__shared__ int32_t sh_agg_dst_ids[SH_MEM_SIZE];
	__shared__ int32_t sh_sampled_id[SH_MEM_SIZE];
	/*local offset: 0, local node offset; 1, global node offset; 2, local edge offset; 3, global edge offset */
	__shared__ int32_t local_offset[4];
	int32_t* input_ids = nullptr;
	int32_t batch_size = 0;
	if(op_id == INTRABATCH_CON){
		input_ids = sampled_ids;
		batch_size = node_counter[1];
	}else if(op_id > INTRABATCH_CON){
		input_ids = agg_src_ids + edge_counter[0];
		batch_size = edge_counter[1];
	}
	for(int32_t idx = threadIdx.x + blockDim.x * blockIdx.x; idx < batch_size * count; idx += gridDim.x * blockDim.x){
		if(threadIdx.x == 0){
            local_offset[0] = 0;
			local_offset[1] = 0;
            local_offset[2] = 0;
            local_offset[3] = 0;
        }
        __syncthreads();
		int32_t sample_src_id = input_ids[idx/count];
		int32_t sample_dst_id;
		if(sample_src_id >= 0){
			int32_t neighbor_offset = idx%count;
			int64_t start_index = csr_node_index[sample_src_id];
			int32_t col_size = csr_node_index[(sample_src_id + 1)] - start_index;
			if(neighbor_offset >= col_size){
				sample_dst_id = -1;
			}else{
				thrust::minstd_rand engine;
				engine.discard(idx);
				thrust::uniform_int_distribution<> dist(0, col_size - 1);
				int32_t dst_index = dist(engine);
				sample_dst_id = csr_dst_node_ids[(int64_t(start_index + int64_t(dst_index)))];
				if(sample_dst_id >= 0){
					atomicAdd(edge_access_time + sample_src_id, 1);

					int32_t bitmap_idx = sample_dst_id / 32;
					int32_t bitmap_off = sample_dst_id % 32;
					uint32_t bitmap_data = (1 << bitmap_off);
					uint32_t old_bitmap_data = atomicOr(accessed_map + bitmap_idx, bitmap_data);
					uint32_t is_accessed = (old_bitmap_data >> bitmap_off) % 2;
					// int32_t acc_count = atomicAdd(accessed_map + sample_dst_id, 1);
					// int32_t acc_count = atomicAdd(accessed_map + sample_dst_id, 1);
					if(is_accessed == 0){//first time node is sampled
						int32_t node_off = atomicAdd(local_offset, 1);
						sh_sampled_id[node_off] = sample_dst_id;
					}
					int32_t edge_off = atomicAdd(local_offset + 2, 1);
					sh_agg_src_ids[edge_off] = sample_dst_id;
					sh_agg_dst_ids[edge_off] = sample_src_id;
				}
			}
		}
		__syncthreads();
		if(threadIdx.x == 0){
            local_offset[1] = atomicAdd(node_counter + INTRABATCH_CON * 2, local_offset[0]);//global node count current hop
			local_offset[3] = atomicAdd(edge_counter + 2, local_offset[2]);//global edge count current hop
		}
        __syncthreads();
		if(threadIdx.x < local_offset[0]){
			int32_t node_base = local_offset[1] + node_counter[0] + node_counter[1];
			int32_t dst_id = sh_sampled_id[threadIdx.x]; 
			sampled_ids[node_base + threadIdx.x] = dst_id;
			position_map[dst_id] = node_base + threadIdx.x;
		}
		__syncthreads();
		if(threadIdx.x < local_offset[2]){
			int32_t edge_base = local_offset[3] + edge_counter[0] + edge_counter[1];
			agg_src_ids[edge_base + threadIdx.x] = sh_agg_src_ids[threadIdx.x];
			agg_dst_ids[edge_base + threadIdx.x] = sh_agg_dst_ids[threadIdx.x]; 
		}
		__syncthreads();
	}
}


extern "C"											
void RandomSample(
  cudaStream_t    strm_hdl, 
  GraphStorage*   graph,
  UnifiedCache*   cache,
  MemoryPool*     memorypool,
  int32_t         count,
  int32_t         dev_id,
  int32_t         op_id,
  bool            is_presc) 
{		
	if (!is_presc) {
		//printf("\t[Info] device-%d RandomSample\n", dev_id);
	}

	if(graph == nullptr){
		std::cout<<"invalid storage ptr\n";
		return;
	}

	int32_t** csr_dst_node_ids 		= graph -> GetCSRNodeMatrix(dev_id);
	int64_t** csr_node_index  		= graph -> GetCSRNodeIndex(dev_id);
	int32_t partition_count 		= graph -> GetPartitionCount();
	char* partition_index 			= graph -> PartitionIndex(dev_id);
	int32_t* parition_offset 		= graph -> PartitionOffset(dev_id);

	uint32_t* accessed_map 			= memorypool->GetAccessedMap();
	int32_t* position_map 			= memorypool->GetPositionMap();
	int32_t* node_counter 			= memorypool->GetNodeCounter();
	int32_t* edge_counter 			= memorypool->GetEdgeCounter();
	int32_t* sampled_ids 			= memorypool->GetSampledIds();
	int32_t* agg_src_ids 			= memorypool->GetAggSrcId();
	int32_t* agg_dst_ids 			= memorypool->GetAggDstId();
	int32_t* agg_src_off 			= memorypool->GetAggSrcOf();
	int32_t* agg_dst_off 			= memorypool->GetAggDstOf();
	char* tmp_partition_index  		= memorypool->GetTmpPartIdx();
	int32_t* tmp_parition_offset	= memorypool->GetTmpPartOff();
	int32_t* cache_index 			= memorypool->GetCacheSearchBuffer();
	int32_t* full_cache_index 	= memorypool->GetFullCacheSearchBuffer();

    dim3 block_num(16, 1);
    dim3 thread_num(OP_THREAD_NUM, 1);
	if(!is_presc){
		int32_t* h_node_counter = (int32_t*)malloc(16*sizeof(int32_t));
		cudaMemcpy(h_node_counter, node_counter, 64, cudaMemcpyDeviceToHost);
		cudaCheckError();

		int32_t* h_edge_counter = (int32_t*)malloc(16*sizeof(int32_t));
		cudaMemcpy(h_edge_counter, edge_counter, 64, cudaMemcpyDeviceToHost);
		cudaCheckError();

		int32_t* input_ids = nullptr;
		int32_t batch_size = 0;
		if(op_id == INTRABATCH_CON){
			input_ids = sampled_ids;
			batch_size = h_node_counter[1];
		}else if(op_id > INTRABATCH_CON){
			input_ids = agg_src_ids + h_edge_counter[0];
			batch_size = h_edge_counter[1];
		}
		cache -> FindTopo(input_ids, tmp_partition_index, tmp_parition_offset, batch_size, op_id, strm_hdl, dev_id);
		cudaCheckError();
		free(h_node_counter);
		free(h_edge_counter);
		random_sample<<<block_num, thread_num, 0, (strm_hdl)>>>(sampled_ids, op_id, csr_node_index, csr_dst_node_ids, 
																											tmp_partition_index, tmp_parition_offset,
																											count, partition_count,
																											agg_src_ids, agg_dst_ids,
																											accessed_map,
																											position_map,
																											node_counter,
																											edge_counter, 
																											dev_id);	
		cudaCheckError();
	}else{
		int32_t* pin_csr_dst_node_ids = graph -> GetCSRNodeMatrixCPU();
		int64_t* pin_csr_node_index  = graph -> GetCSRNodeIndexCPU();
		unsigned long long int* edge_access_time = cache->GetEdgeAccessedMap(dev_id);
		pre_sample<<<block_num, thread_num, 0, (strm_hdl)>>>(sampled_ids, op_id, pin_csr_node_index, pin_csr_dst_node_ids, 
																											partition_index, parition_offset,
																											count, partition_count,
																											agg_src_ids, agg_dst_ids,
																											accessed_map,
																											position_map,
																											node_counter,
																											edge_counter, 
																											dev_id,
																											edge_access_time);	
	}
	
	cudaCheckError();
	construct_graph<<<block_num, thread_num, 0, (strm_hdl)>>>(agg_src_ids, agg_dst_ids, agg_src_off, agg_dst_off,
															  position_map, edge_counter, node_counter, op_id, dev_id, is_presc);
	cudaCheckError();

	/*
	cudaDeviceSynchronize();
	if (!is_presc && dev_id == 0){
		printf("[Info] 输出第 %d 次采样mapping\n", op_id/3);
		
		thrust::device_vector<int32_t> d_edge_counter(edge_counter, edge_counter + 16);
		thrust::host_vector<int32_t> h_edge_counter = d_edge_counter;

		int32_t edge_num = h_edge_counter[2];
		int32_t edge_off = h_edge_counter[0] + h_edge_counter[1];

		thrust::device_vector<int32_t> d_src_ids(agg_src_ids + edge_off, agg_src_ids + edge_off + edge_num);
		thrust::device_vector<int32_t> d_dst_ids(agg_dst_ids + edge_off, agg_dst_ids + edge_off + edge_num);
		thrust::device_vector<int32_t> d_src_off(agg_src_off + edge_off, agg_src_off + edge_off + edge_num);
		thrust::device_vector<int32_t> d_dst_off(agg_dst_off + edge_off, agg_dst_off + edge_off + edge_num);

		thrust::host_vector<int32_t> h_src_ids = d_src_ids;
		thrust::host_vector<int32_t> h_dst_ids = d_dst_ids;
		thrust::host_vector<int32_t> h_src_off = d_src_off;
		thrust::host_vector<int32_t> h_dst_off = d_dst_off;

		for (int32_t i = 0; i < edge_num; i++) {
			printf("\tsrc: %d --> %d, dst: %d ---> %d\n", h_src_off[i], h_src_ids[i], h_dst_off[i], h_dst_ids[i]);
		}
	}
	*/

	counter_update<<<1, 1, 0, (strm_hdl)>>>(node_counter, edge_counter, op_id, 0, 0);		
	cudaCheckError();	

	if(!is_presc){
		cache->FindFeat(sampled_ids, cache_index, node_counter, op_id, strm_hdl, dev_id);

		// 当前 hop 的 cache_index 结果保存
		int32_t* h_node_counter = (int32_t*)malloc(64);
		cudaMemcpy(h_node_counter, node_counter, 64, cudaMemcpyDeviceToHost);
		int32_t node_off = h_node_counter[(op_id % INTRABATCH_CON) * 2];
		int32_t batch_size = h_node_counter[(op_id % INTRABATCH_CON) * 2 + 1];
		
		//===== debug
		//thrust::device_vector<int32_t> d_cache_index(cache_index, cache_index + batch_size);
		//thrust::host_vector<int32_t> h_cache_index = d_cache_index;
		//std::printf("random_sample: cache_index[%d~%d]: { ", node_off, node_off + batch_size);
		//for (int i = 0; i < batch_size; i++) {
		//	std::printf("%d, ", h_cache_index[i]);
		//}
		//std::printf("}\n");
		//===== debug

		dim3 ffc_block(16, 1, 1); 
		dim3 ffc_thread(OP_THREAD_NUM, 1);	
		fillFullCacheSearchBuffer<<<ffc_block, ffc_thread, 0, (strm_hdl)>>>(cache_index, full_cache_index, node_off, batch_size, op_id, dev_id);

		//thrust::copy(thrust::cuda::par.on(strm_hdl), cache_index, cache_index + batch_size, full_cache_index + node_off);
		cudaCheckError();
	}

	/*
	if (!is_presc) {
		if (dev_id == 0) {
			thrust::device_vector<int32_t> d_edge_counter(edge_counter, edge_counter + 16);
			thrust::host_vector<int32_t> h_edge_counter = d_edge_counter;
			int32_t one_hop_edge_num = h_edge_counter[10];
			int32_t two_hop_edge_num = h_edge_counter[11];

			int32_t start, end, edge_num;
			if (op_id == 3) {
				start = 0;
				end = one_hop_edge_num;
				edge_num = one_hop_edge_num;
			} else {
				start = one_hop_edge_num;
				end = two_hop_edge_num;
				edge_num = two_hop_edge_num;
			}

			//std::printf("[Info] start: %d, end: %d, edge_num: %d\n", start, end, edge_num);

			thrust::device_vector<int32_t> d_src(agg_src_ids + start, agg_src_ids + end);
			thrust::device_vector<int32_t> d_dst(agg_dst_ids + start, agg_dst_ids + end);
			thrust::host_vector<int32_t> h_src = d_src;
			thrust::host_vector<int32_t> h_dst = d_dst;

			std::printf("[Info] random sample(%d), edge_num: %d\n", op_id, edge_num);
			
			std::printf("src,dst: \n");
			for(int32_t i = 0; i < h_src.size(); i++){
				printf("\t%d, %d\n", h_src[i], h_dst[i]);
			}
			printf("\n\n");
			
			std::printf("sorted src,dst: \n");
			thrust::sort(
				thrust::host,
	 			thrust::make_zip_iterator(thrust::make_tuple(h_src.begin(), h_dst.begin())),
	 			thrust::make_zip_iterator(thrust::make_tuple(h_src.end(), h_dst.end())),
				[] __host__ (const auto& a, const auto& b) {
					if (thrust::get<0>(a) != thrust::get<0>(b))
						return thrust::get<0>(a) < thrust::get<0>(b);
					return thrust::get<1>(a) < thrust::get<1>(b);
				}
			);
			for(int32_t i = 0; i < h_src.size(); i++){
				printf("\t%d, %d\n", h_src[i], h_dst[i]);
			}
			printf("\n\n");
		}
	}
	*/

	/*
	if (!is_presc) {
		if (dev_id == 0) {
			thrust::device_vector<int32_t> d_edge_counter(edge_counter, edge_counter + 16);
			thrust::host_vector<int32_t> h_edge_counter = d_edge_counter;
			int32_t one_hop_edge_num = h_edge_counter[10];
			int32_t two_hop_edge_num = h_edge_counter[11];

			int32_t start, end, edge_num;
			if (op_id == 3) {
				start = 0;
				end = one_hop_edge_num;
				edge_num = one_hop_edge_num;
			} else {
				start = one_hop_edge_num;
				end = two_hop_edge_num;
				edge_num = two_hop_edge_num;
			}

			//std::printf("[Info] start: %d, end: %d, edge_num: %d\n", start, end, edge_num);

			//thrust::device_vector<int32_t> d_src_off(agg_src_off + start, agg_src_off + end);
			//thrust::device_vector<int32_t> d_dst_off(agg_dst_off + start, agg_dst_off + end);
			thrust::device_vector<int32_t> d_src_off(agg_dst_off + start, agg_dst_off + end);
			thrust::device_vector<int32_t> d_dst_off(agg_src_off + start, agg_src_off + end);
			thrust::host_vector<int32_t> h_src_off = d_src_off;
			thrust::host_vector<int32_t> h_dst_off = d_dst_off;

			std::printf("[Info] random sample(%d), edge_num: %d\n", op_id, edge_num);
			
			std::printf("src_off,dst_off: \n");
			for(int32_t i = 0; i < h_src_off.size(); i++){
				printf("\t%d, %d\n", h_src_off[i], h_dst_off[i]);
			}
			printf("\n\n");
			
			std::printf("sorted src_off,dst_off: \n");
			thrust::sort(
				thrust::host,
	 			thrust::make_zip_iterator(thrust::make_tuple(h_src_off.begin(), h_dst_off.begin())),
	 			thrust::make_zip_iterator(thrust::make_tuple(h_src_off.end(), h_dst_off.end())),
				[] __host__ (const auto& a, const auto& b) {
					if (thrust::get<0>(a) != thrust::get<0>(b))
						return thrust::get<0>(a) < thrust::get<0>(b);
					return thrust::get<1>(a) < thrust::get<1>(b);
				}
			);
			for(int32_t i = 0; i < h_src_off.size(); i++){
				printf("\t%d, %d\n", h_src_off[i], h_dst_off[i]);
			}
			printf("\n\n");
		}
	}
	*/

	/*
	if (!is_presc && dev_id == 0 && op_id == 6){
		int* h_edge = new int32_t[16];;
		cudaMemcpy(h_edge, edge_counter, 16 * sizeof(int32_t), cudaMemcpyDeviceToHost);
		printf("cpu edge counter info:\n");
		for (int i = 0; i < 16; i++){
			printf("%d ", h_edge[i]);
		}
		int h_edge_count;
		cudaMemcpy(&h_edge_count, edge_counter+11, sizeof(int32_t), cudaMemcpyDeviceToHost);
		printf(">>>>>>>>h_edge_count in G0 :%d\n",h_edge_count);
		int32_t* h_agg_src_ids = (int32_t*)malloc(h_edge_count*sizeof(int32_t));
		cudaMemcpy(h_agg_src_ids, agg_src_ids, h_edge_count*sizeof(int32_t), cudaMemcpyDeviceToHost);
		int32_t* h_agg_dst_ids = (int32_t*)malloc(h_edge_count*sizeof(int32_t));
		cudaMemcpy(h_agg_dst_ids, agg_dst_ids, h_edge_count*sizeof(int32_t), cudaMemcpyDeviceToHost);
		
		std::vector<int> indices;
		indices.resize(h_edge_count);

		for (int i = 0; i < h_edge_count; ++i) {
			indices[i] = i;
		}

		std::sort(indices.begin(), indices.end(), [&](int a, int b) {
			if (h_agg_src_ids[a] != h_agg_src_ids[b]) {
				return h_agg_src_ids[a] < h_agg_src_ids[b]; // 先按行排序
			} else {
				return h_agg_dst_ids[a] < h_agg_dst_ids[b]; // 行相同则按列排序
			}
		});

		// 根据排序后的索引重新排列原数组
		std::vector<int> sorted_src;
		sorted_src.resize(h_edge_count);
		std::vector<int> sorted_dst;
		sorted_dst.resize(h_edge_count);
		for (int i = 0; i < h_edge_count; ++i) {
			sorted_src[i] = h_agg_src_ids[indices[i]];
			sorted_dst[i] = h_agg_dst_ids[indices[i]];
		}

		printf(">>> random sample result:\n");
		for(int i = 0; i < h_edge_count; i ++){
			printf("%d, %d\n", sorted_src[i], sorted_dst[i]);
		}

	}
	if (!is_presc && op_id == 6){
		while(1){}
	}
	*/
}

extern "C"
void FeatureCacheLookup(
  cudaStream_t    strm_hdl,
  UnifiedCache*   cache, 
  MemoryPool*     memorypool,
  int32_t         op_id,
  int32_t         dev_id)
{	
	int32_t* sampled_ids 		= memorypool->GetSampledIds();
	int32_t* cache_index 		= memorypool->GetCacheSearchBuffer();
	float* dst_float_buffer 	= memorypool->GetFloatFeatures();
	int32_t* node_counter 		= memorypool->GetNodeCounter();
	int32_t* edge_counter 		= memorypool->GetEdgeCounter();

	counter_update<<<1, 1, 0, (strm_hdl)>>>(node_counter, edge_counter, op_id, 0, 0);		
	cudaCheckError();
	cache->FeatCacheLookup(sampled_ids, cache_index, node_counter, dst_float_buffer, op_id, dev_id, strm_hdl);
	cudaCheckError();
}	

extern "C"
void IOSubmit(
	cudaStream_t    strm_hdl, 
	FeatureStorage* feature,
  	MemoryPool*     memorypool,
	int32_t			op_id,
	int32_t         dev_id)
{
	// int32_t* sampled_ids 		= memorypool->GetSampledIds();
	// int32_t* cache_index 		= memorypool->GetCacheSearchBuffer();
	// float* dst_float_buffer 	= memorypool->GetFloatFeatures();
	// int32_t* node_counter 		= memorypool->GetNodeCounter();
	// int32_t* edge_counter 		= memorypool->GetEdgeCounter();

	// counter_update<<<1, 1, 0, (strm_hdl)>>>(node_counter, edge_counter, op_id, 0, 0);		
	// cudaCheckError();
	// // feature->IOSubmit(sampled_ids, cache_index, node_counter, dst_float_buffer, op_id, dev_id, strm_hdl);
	// // cudaCheckError();
}


__global__ void ClearPosMap(int32_t* position_map, int32_t* sampled_ids, int32_t* node_counter){
	int32_t batch_size = node_counter[INTRABATCH_CON * 2 + 1];
	for(int32_t idx = threadIdx.x + blockDim.x * blockIdx.x; idx < batch_size; idx += gridDim.x * blockDim.x){
		int32_t sampled_id = sampled_ids[idx];
		position_map[sampled_id] = 0;
	}
}

extern "C"
void IOComplete(
  cudaStream_t    strm_hdl, 
  UnifiedCache*   cache, 
  MemoryPool*     memorypool,
  int32_t         dev_id,
  int32_t         mode)
{
	//printf("\t[Info] IOComplete\n");

	if(mode == TRAINMODE){
		int32_t* agg_dst_id = memorypool->GetAggDstId();
		int32_t* agg_src_id = memorypool->GetAggSrcId();
		int32_t* node_counter = memorypool->GetNodeCounter();
		int32_t* edge_counter = memorypool->GetEdgeCounter();
		int32_t* sampled_ids = memorypool->GetSampledIds();
		int32_t* agg_src_off = memorypool->GetAggSrcOf();
		int32_t* agg_dst_off = memorypool->GetAggDstOf();

		int32_t* position_map = memorypool->GetPositionMap();
		// int32_t* h_node_counter = (int32_t*)malloc(64);
		// cudaDeviceSynchronize();
		// cudaMemcpy(h_node_counter, edge_counter, 64, cudaMemcpyDeviceToHost);
		// for(int i = 0; i < 16; i++){
		// 	std::cout<<i<<" h c"<<h_node_counter[i]<<" on "<<dev_id<<"\n";
		// }
			// cudaMemsetAsync(position_map, 0, int64_t(int64_t(total_node_num) * int64_t(sizeof(int32_t))), (strm_hdl));
		dim3 block_num(32, 1);
		dim3 thread_num(OP_THREAD_NUM, 1);
		ClearPosMap<<<block_num, thread_num, 0, strm_hdl>>>(position_map, sampled_ids, node_counter);
		cache -> CacheProfiling(sampled_ids, agg_src_id, agg_dst_id, agg_src_off, agg_dst_off, node_counter, edge_counter, strm_hdl, dev_id);
	}
}

int local_step = 0;

void formatGenerate(
	int32_t *raw_full_cache_index,
	thrust::device_ptr<vint> &row_indices,
	thrust::device_ptr<vint> &col_indices,
	vint num_rows,
	vint num_cols,
	int32_t nnz,
	IPCEnv *env,
	int32_t dev_id,
	int32_t current_pipe,
	MemoryPool*     memorypool
) {
	#define output_sample_result_to_mtx
	#ifdef output_sample_result_to_mtx
	// 复制到 Host
    thrust::host_vector<vint> h_row(nnz);
    thrust::host_vector<vint> h_col(nnz);
    thrust::copy(row_indices, row_indices + nnz, h_row.begin());
    thrust::copy(col_indices, col_indices + nnz, h_col.begin());

    // 生成随机权重 [-20, 20]
    std::mt19937 rng(42);  // 固定种子方便复现
    std::uniform_real_distribution<float> dist(-20.0f, 20.0f);

    // 打开输出文件
	std::string filename = "/home/gaohy/Legions/TAC/result/products_batch_800/sample-data/" + std::to_string(dev_id) + "-" + std::to_string(local_step++) + ".mtx";
    std::ofstream fout(filename);
    if (!fout.is_open()) {
        std::cerr << "Failed to open file: " << filename << "\n";
        return;
    }

    // 写头部
    fout << "%%MatrixMarket matrix coordinate real general\n";
    fout << "%-----------------------------------------------------------------------------------\n";
    fout << (num_rows) << " " << (num_cols) << " " << nnz << "\n";  // 1-based indexing

    // 写数据
    for (int i = 0; i < nnz; ++i) {
        // Matrix Market 是 1-based，所以加 1
        int row = h_row[i] + 1;
        int col = h_col[i] + 1;
        float val = dist(rng);
        fout << row << " " << col << " " << std::fixed << std::setprecision(4) << val << "\n";
    }

    fout.close();	
	#endif
	#undef output_sample_result_to_mtx

	// [预备工作] 采样子图冷热数据标识生成（目前仅列重排布划分冷热数据，行重排可以增强 denseB 的访问局部性，但相似度计算目前开销较大，暂不进行行重排）
	// cache_index 是 node_map_ 哈希表的输出结果，标识了不同 sample id 是否缓存在 gpu 上（不在 gpu 时为负数）
    thrust::device_vector<bool> hotFlag(nnz);
	thrust::transform(
        thrust::device,                     // 在设备端执行
        col_indices,                // 输入起始位置
        col_indices + nnz,                  // 输入结束位置
        hotFlag.begin(),                    // 输出起始位置
        [=] __device__ (int32_t v) {
            return raw_full_cache_index[v] >= 0;       // v 是 col_indices 的元素值
        }
    );

	// 冷热数据划分（对于 [0, first_randomsample_nnz) 范围内的边进行冷热数据划分）
	thrust::sort(
		thrust::device,
        thrust::make_zip_iterator(thrust::make_tuple(hotFlag.begin(), row_indices, col_indices)),               
        thrust::make_zip_iterator(thrust::make_tuple(hotFlag.end(), row_indices + nnz, col_indices + nnz)),     
        [] __device__ (const thrust::tuple<bool, vint, vint>& a,
                                const thrust::tuple<bool, vint, vint>& b) {
            // 1. hotFlag 降序（true 在前）
            if (thrust::get<0>(a) != thrust::get<0>(b)) {
                return thrust::get<0>(a) > thrust::get<0>(b);
            }
            // 2. row_indices 升序（小的在前）
            if (thrust::get<1>(a) != thrust::get<1>(b)) {
                return thrust::get<1>(a) < thrust::get<1>(b);
            }
            // 3. col_indices 升序（小的在前）
            return thrust::get<2>(a) < thrust::get<2>(b);
        }
    );
    // returns an iterator pointing to the end of the true partition of a partitioned range(sync, although we specify the stream)
    auto boundary = thrust::partition_point(
		thrust::device,
        hotFlag.begin(), hotFlag.end(),
        [] __device__ (bool flag) { return flag == true; }
    );
    vint hot_count = boundary - hotFlag.begin(); // 热数据边数

	// 负责生成 BCSR 和 BCSR 的 coo 只负责把前期的采样结果进行结构化组织，因此其不具有数据所有权，不管理数据的生命周期
    COO<MAT_VAL_TYPE> hotCoo(num_rows, num_cols, hot_count, thrust::raw_pointer_cast(row_indices), thrust::raw_pointer_cast(col_indices));
    COO<MAT_VAL_TYPE> coldCoo(num_rows, num_cols, nnz - hot_count, thrust::raw_pointer_cast(row_indices) + hot_count, thrust::raw_pointer_cast(col_indices) + hot_count);

	// bcsr & bcsc 构造
	vint* rowWindowOffset = memorypool->GetBCSRRowWindowOffset();
	vint* tcOffset_bcsr = memorypool->GetBCSRTcOffset();
	vint* sparseA2B = memorypool->GetBCSRSparseA2B();
	TCLOCAL_TYPE* tcLocalBit_bcsr = memorypool->GetBCSRTcLocalBit();

	vint* colWindowOffset = memorypool->GetBCSCColWindowOffset();
	vint* tcOffset_bcsc = memorypool->GetBCSCTcOffset();
	vint* sparseA2C = memorypool->GetBCSCSparseA2C();
	TCLOCAL_TYPE* tcLocalBit_bcsc = memorypool->GetBCSCTcLocalBit();

	auto start = std::chrono::high_resolution_clock::now();
	//cudaProfilerStart();
	BCSR<MAT_VAL_TYPE> bcsr(hotCoo, rowWindowOffset, tcOffset_bcsr, sparseA2B, tcLocalBit_bcsr, memorypool->GetCurrentPipe());
    BCSC<MAT_VAL_TYPE> bcsc(coldCoo, colWindowOffset, tcOffset_bcsc, sparseA2C, tcLocalBit_bcsc, memorypool->GetCurrentPipe());
	//cudaProfilerStop();
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	if (dev_id == 0) {
    	std::cout << "BCSR & BCSC 生成耗时: " << duration.count() << " ms" << std::endl;
	}

	// IPC 空间获取，使训练进程获取到结构体数据
	env->FillUpBCSRBuffer(
		bcsr.num_rows, 
		bcsr.num_cols, 
		bcsr.nnz, 
		bcsr.num_TCBlocks, 
		bcsr.num_rowWindows,
		dev_id,
		current_pipe
	);
	env->FillUpBCSCBuffer(
		bcsc.num_rows, 
		bcsc.num_cols, 
		bcsc.nnz, 
		bcsc.num_TCBlocks, 
		bcsc.num_colWindows,
		dev_id,
		current_pipe
	);
}

extern "C"
void FormatConvert(
  cudaStream_t    strm_hdl,
  IPCEnv *env,
  MemoryPool*     memorypool,
  int32_t         dev_id)
{	
	auto start = std::chrono::high_resolution_clock::now();
	// [预备工作] row_indices 和 col_indices 均可保证非负
	int32_t *raw_sampled_ids = memorypool->GetSampledIds();
	// 这里的方案要对应原始数据的格式以及 random_sample 处的处理方式
	vint *raw_row_indices = reinterpret_cast<vint*>(memorypool->GetAggDstOf());
	vint *raw_col_indices = reinterpret_cast<vint*>(memorypool->GetAggSrcOf());
	int32_t *raw_node_counter = memorypool->GetNodeCounter();
	int32_t *raw_edge_counter = memorypool->GetEdgeCounter();
	int32_t *raw_full_cache_index = memorypool->GetFullCacheSearchBuffer(); // cache_index 的顺序同 sampled_ids 的顺序
	int32_t current_pipe = memorypool->GetCurrentPipe();

	// [预备工作] raw pointer -> thrust device pointer，便于使用 thrust API
	auto sampled_ids= thrust::device_pointer_cast(raw_sampled_ids);
	auto row_indices = thrust::device_pointer_cast(raw_row_indices);
	auto col_indices = thrust::device_pointer_cast(raw_col_indices);
	auto node_counter= thrust::device_pointer_cast(raw_node_counter);
	auto edge_counter= thrust::device_pointer_cast(raw_edge_counter);
	auto full_cache_index= thrust::device_pointer_cast(raw_full_cache_index);

	// [预备工作] D2H 拷贝计数器
	thrust::host_vector<int32_t> h_node_counter(node_counter, node_counter + 16);
	thrust::host_vector<int32_t> h_edge_counter(edge_counter, edge_counter + 16);

	// [预备工作] 属性获取
	int32_t batch_node_num = h_node_counter[9];					// batch generate 节点数
	int32_t first_randomsample_node_num = h_node_counter[10];	// batch generate + 1hop 邻居总节点数
	int32_t all_node_num = h_node_counter[11];					// batch generate 和 一二阶邻居总节点数

	int32_t first_randomsample_nnz = h_edge_counter[10]; 		// 一阶邻居总边数
	int32_t all_nnz = h_edge_counter[11]; 						// 一阶邻居和二阶邻居总边数

	// [预备工作] 添加反向边
	//std::printf("[Info] device %d 增加反向边\n", dev_id);
	thrust::device_vector<vint> row_indices_add_reverse(all_nnz * 2);
	thrust::device_vector<vint> col_indices_add_reverse(all_nnz * 2);

	// row_indices
	thrust::copy(thrust::device, row_indices, row_indices + all_nnz, row_indices_add_reverse.begin());
	thrust::copy(thrust::device, col_indices, col_indices + all_nnz, row_indices_add_reverse.begin() + all_nnz);
	// col_indices
	thrust::copy(thrust::device, col_indices, col_indices + all_nnz, col_indices_add_reverse.begin());
	thrust::copy(thrust::device, row_indices, row_indices + all_nnz, col_indices_add_reverse.begin() + all_nnz);

	// [预备工作] 重复边去重(需要注意要同时修改 counter 的数值，否则训练过程获取数据时会出错)
	// 1. 先按 (row_indices, col_indices) 排序，确保相同 row_indices 的 col_indices 也是有序的
	thrust::sort(
		thrust::device,
        thrust::make_zip_iterator(thrust::make_tuple(row_indices_add_reverse.begin(), col_indices_add_reverse.begin())),               
        thrust::make_zip_iterator(thrust::make_tuple(row_indices_add_reverse.end(), col_indices_add_reverse.end())),     
		[] __device__ (const thrust::tuple<vint, vint>& a, const thrust::tuple<vint, vint>& b) {
			if (thrust::get<0>(a) != thrust::get<0>(b)) {
				return thrust::get<0>(a) < thrust::get<0>(b);
			}
			return thrust::get<1>(a) < thrust::get<1>(b);
		}
	);
	auto zipped_edge_begin = thrust::make_zip_iterator(
		thrust::make_tuple(row_indices_add_reverse.begin(), col_indices_add_reverse.begin()));

	auto zipped_edge_end = thrust::make_zip_iterator(
		thrust::make_tuple(row_indices_add_reverse.end(), col_indices_add_reverse.end()));

	// 2. 使用 unique 去重，使得相同 row_indices 下的 col_indices 只保留唯一值
	auto zipped_new_edge_end = thrust::unique(
		thrust::device,        											// 显式指定执行策略
		zipped_edge_begin,         							// 起始迭代器
		zipped_edge_end,         							// 起始迭代器
		[] __device__ (const auto& a, const auto& b) {
			return thrust::get<0>(a) == thrust::get<0>(b) &&  // 比较行索引
				   thrust::get<1>(a) == thrust::get<1>(b);     // 比较列索引
		}         										// lambda比较器
	);

	// 3. 计算去重后的长度
	int new_edge_size_add_reverse = zipped_new_edge_end - zipped_edge_begin;
	row_indices_add_reverse.resize(new_edge_size_add_reverse);
	col_indices_add_reverse.resize(new_edge_size_add_reverse);

	// ============= 全图构造 =============
	row_indices = row_indices_add_reverse.data();
	col_indices = col_indices_add_reverse.data();

	int32_t num_rows = all_node_num;
	int32_t num_cols = all_node_num;
	int32_t nnz = new_edge_size_add_reverse;

	auto start_formatGenerate = std::chrono::high_resolution_clock::now();	
	formatGenerate(
		raw_full_cache_index, 
		row_indices, col_indices,
		num_rows, num_cols, nnz, 
		env, dev_id, current_pipe, 
		memorypool
	);
	auto end_formatGenerate = std::chrono::high_resolution_clock::now();
	auto duration_formatGenerate = std::chrono::duration_cast<std::chrono::milliseconds>(end_formatGenerate - start_formatGenerate);
	//if (dev_id == 0) {
 //   	std::cout << "预处理及格式生成总耗时: " << duration_formatGenerate.count() << " ms" << std::endl;
	//}

	// ============= 原始采样结果添加反向边 =============
	cudaMemcpy(
		raw_row_indices,
		thrust::raw_pointer_cast(row_indices),
		nnz * sizeof(vint),
		cudaMemcpyDeviceToDevice
	);

	cudaMemcpy(
		raw_col_indices,
		thrust::raw_pointer_cast(col_indices),
		nnz * sizeof(vint),
		cudaMemcpyDeviceToDevice
	);

	cudaMemcpy(&raw_edge_counter[10], &nnz, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(&raw_edge_counter[11], &nnz, sizeof(int), cudaMemcpyHostToDevice);

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

	//if (dev_id == 0) {
 //		std::cout << "总耗时: " << duration.count() << " ms" << std::endl;
	//}
}	