#include "module.h"

std::vector<void*> MatCs_;

torch::Tensor forward_tensorcore_mixed(
    vint bcsr_num_rows, vint bcsr_num_cols, vint bcsr_nnz, vint bcsr_num_TCBlocks, vint bcsr_num_rowWindows,
    torch::Tensor bcsr_rowWindowOffset, torch::Tensor bcsr_tcOffset, torch::Tensor bcsr_sparseA2B, torch::Tensor bcsr_tcLocalBit, torch::Tensor bcsr_data,
    vint bcsc_num_rows, vint bcsc_num_cols, vint bcsc_nnz, vint bcsc_num_TCBlocks, vint bcsc_num_colWindows,
    torch::Tensor bcsc_colWindowOffset, torch::Tensor bcsc_tcOffset, torch::Tensor bcsc_sparseA2C, torch::Tensor bcsc_tcLocalBit, torch::Tensor bcsc_data,
    int64_t cpu_float_feature_len, int32_t gpu_node_capacity, int32_t feature_dim,
    torch::Tensor id_map, float* cpu_feature_cache_ptr, torch::Tensor cache_search_map, float** gpu_feature_cache_ptr) {

    //std::printf("\n[Info] first layer forward\n"); 

    // bcsr 构造
    vint *raw_bcsr_rowWindowOffset = bcsr_rowWindowOffset.data_ptr<vint>();
    vint *raw_bcsr_tcOffset = bcsr_tcOffset.data_ptr<vint>();
    vint *raw_bcsr_sparseA2B = bcsr_sparseA2B.data_ptr<vint>();
    TCLOCAL_TYPE *raw_bcsr_tcLocalBit = bcsr_tcLocalBit.data_ptr<TCLOCAL_TYPE>();
    MAT_VAL_TYPE *raw_bcsr_data = bcsr_data.data_ptr<MAT_VAL_TYPE>();

    BCSR<MAT_VAL_TYPE> bcsr(
        bcsr_num_rows, bcsr_num_cols, bcsr_nnz, bcsr_num_TCBlocks, bcsr_num_rowWindows, 
        raw_bcsr_rowWindowOffset,
        raw_bcsr_tcOffset,
        raw_bcsr_sparseA2B,
        raw_bcsr_tcLocalBit,
        raw_bcsr_data
    );

    //std::printf("[Info] training-bcsr: \n");
    //std::cout << bcsr.toString() << std::endl;

    // bcsc 构造
    vint *raw_bcsc_colWindowOffset = bcsc_colWindowOffset.data_ptr<vint>();
    vint *raw_bcsc_tcOffset = bcsc_tcOffset.data_ptr<vint>();
    vint *raw_bcsc_sparseA2C = bcsc_sparseA2C.data_ptr<vint>();
    TCLOCAL_TYPE *raw_bcsc_tcLocalBit = bcsc_tcLocalBit.data_ptr<TCLOCAL_TYPE>();
    MAT_VAL_TYPE *raw_bcsc_data = bcsc_data.data_ptr<MAT_VAL_TYPE>();

    BCSC<MAT_VAL_TYPE> bcsc(
        bcsc_num_rows, bcsc_num_cols, bcsc_nnz, bcsc_num_TCBlocks, bcsc_num_colWindows, 
        raw_bcsc_colWindowOffset,
        raw_bcsc_tcOffset,
        raw_bcsc_sparseA2C,
        raw_bcsc_tcLocalBit,
        raw_bcsc_data
    );

    //std::printf("[Info] training-bcsc: \n"); 
    //std::cout << bcsc.toString() << std::endl;

    // spmm
    int32_t* raw_id_map = id_map.data_ptr<int32_t>();
    int32_t* raw_cache_search_map = cache_search_map.data_ptr<int32_t>();
    //float* raw_cpu_feature_cache_ptr = static_cast<float*>(cpu_feature_cache_ptr.get_pointer()); 
    //float** raw_gpu_feature_cache_ptr = static_cast<float**>(gpu_feature_cache_ptr.get_pointer()); 
    float* raw_cpu_feature_cache_ptr = cpu_feature_cache_ptr; 
    float** raw_gpu_feature_cache_ptr = gpu_feature_cache_ptr; 

    MAT_VAL_TYPE *d_MatC;
    launch_spmm_kernel(bcsr, bcsc, raw_id_map, raw_cpu_feature_cache_ptr, raw_cache_search_map, raw_gpu_feature_cache_ptr, d_MatC, gpu_node_capacity, feature_dim); // 目前 denseB 数据还可以模拟生成，后续需要加上 cpu_float_feature 和 gpu_float_feature 数据提取真实 feature 数据

    // denseMatC 计算结果转换为 tensor 数据返回。
    int current_dev = -1;
    cudaGetDevice(&current_dev);
    auto device = "cuda:" + std::to_string(current_dev);

    //std::printf("device-%d\n", current_dev);

    auto output = torch::from_blob(
        d_MatC, 
        {(long long)(bcsr_num_rows) * feature_dim}, 
        torch::TensorOptions().dtype(torch::kFloat32).device(device)
    );

    MatCs_.push_back(d_MatC);

    return output;
}

torch::Tensor forward_tensorcore(
    vint bcsr_num_rows, vint bcsr_num_cols, vint bcsr_nnz, vint bcsr_num_TCBlocks, vint bcsr_num_rowWindows,
    torch::Tensor bcsr_rowWindowOffset, torch::Tensor bcsr_tcOffset, torch::Tensor bcsr_sparseA2B, torch::Tensor bcsr_tcLocalBit, torch::Tensor bcsr_data,
    vint bcsc_num_rows, vint bcsc_num_cols, vint bcsc_nnz, vint bcsc_num_TCBlocks, vint bcsc_num_colWindows,
    torch::Tensor bcsc_colWindowOffset, torch::Tensor bcsc_tcOffset, torch::Tensor bcsc_sparseA2C, torch::Tensor bcsc_tcLocalBit, torch::Tensor bcsc_data,
    int32_t feature_dim, torch::Tensor x) {

    //std::printf("\n[Info] second layer forward\n"); 

    // bcsr 构造
    vint *raw_bcsr_rowWindowOffset = bcsr_rowWindowOffset.data_ptr<vint>();
    vint *raw_bcsr_tcOffset = bcsr_tcOffset.data_ptr<vint>();
    vint *raw_bcsr_sparseA2B = bcsr_sparseA2B.data_ptr<vint>();
    TCLOCAL_TYPE *raw_bcsr_tcLocalBit = bcsr_tcLocalBit.data_ptr<TCLOCAL_TYPE>();
    MAT_VAL_TYPE *raw_bcsr_data = bcsr_data.data_ptr<MAT_VAL_TYPE>();

    BCSR<MAT_VAL_TYPE> bcsr(
        bcsr_num_rows, bcsr_num_cols, bcsr_nnz, bcsr_num_TCBlocks, bcsr_num_rowWindows, 
        raw_bcsr_rowWindowOffset,
        raw_bcsr_tcOffset,
        raw_bcsr_sparseA2B,
        raw_bcsr_tcLocalBit,
        raw_bcsr_data
    );

    //std::printf("[Info] training-bcsr: \n");
    //std::cout << bcsr.toString() << std::endl;

    // bcsc 构造
    vint *raw_bcsc_colWindowOffset = bcsc_colWindowOffset.data_ptr<vint>();
    vint *raw_bcsc_tcOffset = bcsc_tcOffset.data_ptr<vint>();
    vint *raw_bcsc_sparseA2C = bcsc_sparseA2C.data_ptr<vint>();
    TCLOCAL_TYPE *raw_bcsc_tcLocalBit = bcsc_tcLocalBit.data_ptr<TCLOCAL_TYPE>();
    MAT_VAL_TYPE *raw_bcsc_data = bcsc_data.data_ptr<MAT_VAL_TYPE>();

    BCSC<MAT_VAL_TYPE> bcsc(
        bcsc_num_rows, bcsc_num_cols, bcsc_nnz, bcsc_num_TCBlocks, bcsc_num_colWindows, 
        raw_bcsc_colWindowOffset,
        raw_bcsc_tcOffset,
        raw_bcsc_sparseA2C,
        raw_bcsc_tcLocalBit,
        raw_bcsc_data
    );

    //std::printf("[Info] training-bcsc: \n"); 
    //std::cout << bcsc.toString() << std::endl;

    // matB & matC 准备
    MAT_VAL_TYPE *raw_MatB = x.data_ptr<MAT_VAL_TYPE>();
    MAT_VAL_TYPE *d_MatC;

    //thrust::device_vector<MAT_VAL_TYPE> d_MatB(raw_MatB, raw_MatB + bcsc_num_cols * feature_dim);
    //thrust::host_vector<MAT_VAL_TYPE> h_MatB = d_MatB;
    //std::printf("[Info] second layer feature data: { ");
    //for (int32_t i = 0; i < h_MatB.size(); i++) {
    //    if (i && i % 256 == 0) {
    //        std::cout << std::endl;
    //    }
    //    std::cout << h_MatB[i] << ", ";
    //}
    //std::cout << std::endl;

    // spmm
    launch_spmm_kernel(bcsr, bcsc, raw_MatB, d_MatC, feature_dim); // 目前 denseB 数据还可以模拟生成，后续需要加上 cpu_float_feature 和 gpu_float_feature 数据提取真实 feature 数据

    // denseMatC 计算结果转换为 tensor 数据返回。
    int current_dev = -1;
    cudaGetDevice(&current_dev);
    auto device = "cuda:" + std::to_string(current_dev);

    //std::printf("device-%d\n", current_dev);

    auto output = torch::from_blob(
        d_MatC, 
        {(long long)(bcsr_num_rows) * feature_dim}, 
        torch::TensorOptions().dtype(torch::kFloat32).device(device)
    );

    MatCs_.push_back(d_MatC);

    return output;
}

void read_cpu_feature_cache(py::capsule py_cpu_feature_cache_ptr, int32_t cpu_feature_len) {
    float* cpu_feature_cache_ptr = static_cast<float*>(py_cpu_feature_cache_ptr.get_pointer());

    thrust::host_vector<float> h_cpu_feature_cache(cpu_feature_cache_ptr, cpu_feature_cache_ptr + cpu_feature_len);

    std::printf("h_cpu_feature_cache: { ");
    for (int i = 0; i < cpu_feature_len; i++) {
        std::printf("%f, ", h_cpu_feature_cache[i]);
    }
    std::printf("}\n");
}

void read_gpu0_feature_cache(py::capsule py_gpu_feature_cache_ptr) {
    float* gpu_feature_cache_ptr = static_cast<float*>(py_gpu_feature_cache_ptr.get_pointer());
    thrust::device_vector<float> d_gpu_feature_cache(gpu_feature_cache_ptr, gpu_feature_cache_ptr + 100);
    thrust::host_vector<float> h_gpu_feature_cache = d_gpu_feature_cache;

    std::printf("gpu feature: { ");
    for (int i = 0; i < h_gpu_feature_cache.size(); i++) {
        std::printf("%f, ", h_gpu_feature_cache[i]);
    }
    std::printf("}\n");
}

void freeMatC() {
    for (int i = 0; i < MatCs_.size(); i++) {
        cudaFree(MatCs_[i]);
    }
    MatCs_.clear();
}