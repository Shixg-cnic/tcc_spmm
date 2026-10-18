#include "memorypool.cuh"

MemoryPool::MemoryPool(int32_t pipeline_depth){
    pipeline_depth_ = pipeline_depth;
    current_pipe_ = 0;
    sampled_ids_.resize(pipeline_depth_);
    labels_.resize(pipeline_depth_);
    float_features_.resize(pipeline_depth_);
    agg_dst_off_.resize(pipeline_depth_);
    agg_src_off_.resize(pipeline_depth_);
    node_counter_.resize(pipeline_depth_);
    edge_counter_.resize(pipeline_depth_);
    full_cache_search_buffer_.resize(pipeline_depth_);
    bcsr_rowWindowOffset_.resize(pipeline_depth_);
    bcsr_tcOffset_.resize(pipeline_depth_);
    bcsr_sparseA2B_.resize(pipeline_depth_);
    bcsr_tcLocalBit_.resize(pipeline_depth_);
    bcsc_colWindowOffset_.resize(pipeline_depth_);
    bcsc_tcOffset_.resize(pipeline_depth_);
    bcsc_sparseA2C_.resize(pipeline_depth_);
    bcsc_tcLocalBit_.resize(pipeline_depth_);
}
