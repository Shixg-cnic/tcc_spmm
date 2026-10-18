#include "ipcservice.h"
#include "spmm_cnic.h"
#include <cstdio>
#include <vector>
#include <thread>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

#include <nvToolsExt.h>

int32_t hidden_dim = 256;
int32_t num_classes = 47;

#ifdef check
void check_difference(int32_t feat_dim) {
    std::vector<torch::Tensor> sample_data = get_next(feat_dim);
    std::vector<int> block_size = get_block_size();

    torch::Tensor ids = sample_data[0];

    //======================================== 混合数据结构读取 ========================================
    std::vector<vint> sparse_matrix_size = get_sparseMatrix_size();
    std::vector<torch::Tensor> sparse_matrix_data = get_sparseMatrix_data();

    vint f_bcsr_num_rows        = sparse_matrix_size[0];
    vint f_bcsr_num_cols        = sparse_matrix_size[1];
    vint f_bcsr_nnz             = sparse_matrix_size[2];
    vint f_bcsr_num_TCBlocks    = sparse_matrix_size[3];
    vint f_bcsr_num_rowWindows  = sparse_matrix_size[4];

    vint f_bcsc_num_rows        = sparse_matrix_size[5];
    vint f_bcsc_num_cols        = sparse_matrix_size[6];
    vint f_bcsc_nnz             = sparse_matrix_size[7];
    vint f_bcsc_num_TCBlocks    = sparse_matrix_size[8];
    vint f_bcsc_num_colWindows  = sparse_matrix_size[9];

    vint s_bcsr_num_rows        = sparse_matrix_size[10];
    vint s_bcsr_num_cols        = sparse_matrix_size[11];
    vint s_bcsr_nnz             = sparse_matrix_size[12];
    vint s_bcsr_num_TCBlocks    = sparse_matrix_size[13];
    vint s_bcsr_num_rowWindows  = sparse_matrix_size[14];

    vint s_bcsc_num_rows        = sparse_matrix_size[15];
    vint s_bcsc_num_cols        = sparse_matrix_size[16];
    vint s_bcsc_nnz             = sparse_matrix_size[17];
    vint s_bcsc_num_TCBlocks    = sparse_matrix_size[18];
    vint s_bcsc_num_colWindows  = sparse_matrix_size[19];

    torch::Tensor f_bcsr_rowWindowOffset    = sparse_matrix_data[0];
    torch::Tensor f_bcsr_tcOffset           = sparse_matrix_data[1];
    torch::Tensor f_bcsr_sparseA2B          = sparse_matrix_data[2];
    torch::Tensor f_bcsr_tcLocalBit         = sparse_matrix_data[3];
    torch::Tensor f_bcsr_data               = sparse_matrix_data[4];
                                                              
    torch::Tensor f_bcsc_colWindowOffset    = sparse_matrix_data[5];
    torch::Tensor f_bcsc_tcOffset           = sparse_matrix_data[6];
    torch::Tensor f_bcsc_sparseA2C          = sparse_matrix_data[7];
    torch::Tensor f_bcsc_tcLocalBit         = sparse_matrix_data[8];
    torch::Tensor f_bcsc_data               = sparse_matrix_data[9];
                                                              
    torch::Tensor s_bcsr_rowWindowOffset    = sparse_matrix_data[10];
    torch::Tensor s_bcsr_tcOffset           = sparse_matrix_data[11];
    torch::Tensor s_bcsr_sparseA2B          = sparse_matrix_data[12];
    torch::Tensor s_bcsr_tcLocalBit         = sparse_matrix_data[13];
    torch::Tensor s_bcsr_data               = sparse_matrix_data[14];
                                                              
    torch::Tensor s_bcsc_colWindowOffset    = sparse_matrix_data[15];
    torch::Tensor s_bcsc_tcOffset           = sparse_matrix_data[16];
    torch::Tensor s_bcsc_sparseA2C          = sparse_matrix_data[17];
    torch::Tensor s_bcsc_tcLocalBit         = sparse_matrix_data[18];
    torch::Tensor s_bcsc_data               = sparse_matrix_data[19];

    //======================================== feature 信息和数据读取 ========================================
    torch::Tensor cache_search_map = get_memoryaccess_data()[0];
    std::vector<int64_t> feature_attribute = get_feature_attribute();
    std::vector<void*> feature_cache_ptr = get_feature_cache_ptr();

    int64_t cpu_float_feature_len   = feature_attribute[0];
    int64_t gpu_node_capacity       = feature_attribute[1];
    int64_t feature_dim             = feature_attribute[2];

    float* cpu_feature_cache_ptr = static_cast<float*>(feature_cache_ptr[0]);
    float** gpu_feature_cache_ptr = static_cast<float**>(feature_cache_ptr[1]);

    //======================================== 混合访存算子计算 ========================================
    torch::Tensor first_layer_out = forward_tensorcore_mixed(
        s_bcsr_num_rows, s_bcsr_num_cols, s_bcsr_nnz, s_bcsr_num_TCBlocks, s_bcsr_num_rowWindows,
        s_bcsr_rowWindowOffset, s_bcsr_tcOffset, s_bcsr_sparseA2B, s_bcsr_tcLocalBit, s_bcsr_data,
        s_bcsc_num_rows, s_bcsc_num_cols, s_bcsc_nnz, s_bcsc_num_TCBlocks, s_bcsc_num_colWindows,
        s_bcsc_colWindowOffset, s_bcsc_tcOffset, s_bcsc_sparseA2C, s_bcsc_tcLocalBit, s_bcsc_data,
        cpu_float_feature_len, gpu_node_capacity, feature_dim,
        ids, cpu_feature_cache_ptr, cache_search_map, gpu_feature_cache_ptr
    );

    /*
    std::printf("[Info] first layer output:\n");
    torch::Tensor h_first_layer_out = first_layer_out.to(torch::kCPU).contiguous();;
    float* raw_first_layer_out = h_first_layer_out.data_ptr<float>();
    for (int i = 0; i < s_bcsr_num_rows; i++) {
        std::printf("row-%d { ", i);
        for (int j = 0; j < feature_dim; j++) {
            std::printf("%.6f, ", raw_first_layer_out[i * feature_dim + j]);
        }
        std::printf("}\n");
    }
    */
    
    //std::cout << X_out << std::endl;

    //======================================== gpu 上 feature 数据准备 ========================================
    // 构造全 1 feature 矩阵（未指定设备时是分配在 cpu 上的）
    //torch::Tensor x = torch::ones({s_bcsr_num_cols * feature_dim}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    //torch::Tensor x = torch::arange(0, s_bcsr_num_cols * feature_dim, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    //x = x.remainder(100).to(torch::kFloat32) * 0.001 * (torch::where(x % 2 == 1, -1, 1));
    //std::cout << x.device() << std::endl;
    //if (!x.is_contiguous()) {
    //    std::cout << "x is not contiguous" << std::endl;
    //    x = x.contiguous();  // 强制转换为连续存储
    //}

    //bool is_waited = true;
    //while(is_waited);

    // 准备和第一层算子相同的 feature 矩阵
    torch::Tensor h_ids = ids.to(torch::kCPU).contiguous();;
    int32_t* ids_ptr = h_ids.data_ptr<int32_t>();

    float* h_cpu_feature_cache_ptr = static_cast<float*>(get_host_cpu_feature_cache_ptr());

    assert(h_ids.size(0) == s_bcsr_num_cols);
    float* matB = (float*) malloc(sizeof(float) * s_bcsr_num_cols * feature_dim);

    for (int i = 0; i < h_ids.size(0); ++i) {
        vint id = ids_ptr[i];
        for (int j = 0; j < feature_dim; ++j) {
            matB[i * feature_dim + j] = h_cpu_feature_cache_ptr[id * feature_dim + j];
        }
    }

    // 2. 在 GPU 上创建相同形状的 Tensor（使用 torch::from_blob 并不会分配新的内存，所以仍然是 cpu 空间，需要 cudaMemcpy）
    torch::Tensor x = torch::empty({s_bcsr_num_cols * feature_dim}, 
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    cudaMemcpy(x.data_ptr<float>(), matB, sizeof(float) * s_bcsr_num_cols * feature_dim, cudaMemcpyHostToDevice);

    //======================================== 单一访存算子计算 ========================================
    torch::Tensor second_layer_out = forward_tensorcore(
        s_bcsr_num_rows, s_bcsr_num_cols, s_bcsr_nnz, s_bcsr_num_TCBlocks, s_bcsr_num_rowWindows,
        s_bcsr_rowWindowOffset, s_bcsr_tcOffset, s_bcsr_sparseA2B, s_bcsr_tcLocalBit, s_bcsr_data,
        s_bcsc_num_rows, s_bcsc_num_cols, s_bcsc_nnz, s_bcsc_num_TCBlocks, s_bcsc_num_colWindows,
        s_bcsc_colWindowOffset, s_bcsc_tcOffset, s_bcsc_sparseA2C, s_bcsc_tcLocalBit, s_bcsc_data,
        feature_dim, x
    );
    
    /*
    std::printf("[Info] second layer output:\n");
    torch::Tensor h_second_layer_out = second_layer_out.to(torch::kCPU).contiguous();;
    float* raw_second_layer_out = h_second_layer_out.data_ptr<float>();
    for (int i = 0; i < s_bcsr_num_rows; i++) {
        std::printf("row-%d { ", i);
        for (int j = 0; j < feature_dim; j++) {
            std::printf("%.6f, ", raw_second_layer_out[i * feature_dim + j]);
        }
        std::printf("}\n");
    }
    */

    torch::Tensor abs_diff = torch::abs(first_layer_out - second_layer_out);
    float cumulative_diff = abs_diff.sum().item<float>();
    if (cumulative_diff != 0.0f) {
        std::cout << "[Info] Cumulative difference: " << cumulative_diff << std::endl;
    }

    //free(matB);
    freeMatC();
    Synchronize();
}
#endif


int main(int argc, char** argv) {
    // 参数解析
    if (argc != 7) {
        std::cerr << "Usage: " << argv[0] << " <gpu_number>" << std::endl;
        return 1;
    }

    int32_t class_num = std::stoi(argv[1]);
    int32_t feature_dim = std::stoi(argv[2]);
    int32_t hidden_dim = std::stoi(argv[3]);
    int32_t hops_num = std::stoi(argv[4]);
    int32_t gpu_number = std::stoi(argv[5]);
    int32_t epoch = std::stoi(argv[6]);

    //for (int32_t i = 0; i < epoch; i++) {
    //    for (int32_t j = 0; j < gpu_number; ++j) {
    //        std::printf("[Info] GPU %d train\n", j);
    //        cudaSetDevice(j);

    //        InitializeIPC(gpu_number);

    //        std::vector<int32_t> steps = get_steps();
    //        std::printf("train_steps: %d, valid_steps: %d, test_steps: %d\n", steps[0], steps[1], steps[2]);

    //        for (int32_t k = 0; k < steps[0]; k++) {
    //            std::printf("train iter-%d\n", k);
    //            train_one_step(feature_dim);
    //        }
    //        for (int32_t k = 0; k < steps[1]; k++) {
    //            valid_one_step();
    //        }
    //        for (int32_t k = 0; k < steps[2]; k++) {
    //            test_one_step();
    //        }
    //    }
    //}

    // 多线程存在全局变量共享问题(应该隔离)
    //std::vector<std::thread> train_thread_pool;
    //for(int i = 0; i < gpu_number; i++){
    //    std::thread th(&RunnerLoop, i, gpu_number, epoch, feature_dim);
    //    train_thread_pool.push_back(std::move(th));
    //}
    //for(auto &th : train_thread_pool){
    //    th.join();
    //}

    std::vector<pid_t> pids(gpu_number);
    for (int32_t i = 0; i < gpu_number; i++) {
        pids[i] = fork();

        if (pids[i] == 0) {
            //RunnerLoop(i, gpu_number, epoch, feature_dim);

            std::vector<std::string> args_vec = {
                std::to_string(i),
                std::to_string(class_num),
                std::to_string(feature_dim),
                std::to_string(hidden_dim),
                std::to_string(hops_num),
                std::to_string(gpu_number),
                std::to_string(epoch)
            };

            char* args[] = {
                (char*)"/work/gaohy/Legions/TAC/training_backend/c++_frontend/build/worker_with_model",
                (char*)(args_vec[0].c_str()),
                (char*)(args_vec[1].c_str()),
                (char*)(args_vec[2].c_str()),
                (char*)(args_vec[3].c_str()),
                (char*)(args_vec[4].c_str()),
                (char*)(args_vec[5].c_str()),
                (char*)(args_vec[6].c_str()),
                nullptr
            };

            execv(args[0], args);
            perror("execv failed");
            exit(1);

        }
    }

    // 等待所有子进程完成
    for (int32_t i = 0; i < gpu_number; i++) {
        int status;
        waitpid(pids[i], &status, 0);
        if (WIFEXITED(status)) {
            if (!status) {
                std::printf("[Info] GPU %d exited with status %d(normal exit)\n", i, WEXITSTATUS(status));
            } else {
                std::printf("[Error] GPU %d exited abnormally(%d)\n", i, WEXITSTATUS(status));
            }
        } else {
            std::printf("[Error] GPU %d exited abnormally\n", i);
        }
    }
    
    return 0;
}