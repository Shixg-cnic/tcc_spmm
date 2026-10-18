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

void train_one_step(int32_t feat_dim, int device_id, MAT_VAL_TYPE* denseC_first_layer, MAT_VAL_TYPE* denseC_second_layer) {
    torch::Device device(torch::kCUDA);
    
    //======================================== 获取数据 ========================================
    std::string nvtx_msg = "ipc original data";
    nvtxRangePush(nvtx_msg.c_str());
    std::vector<torch::Tensor> sample_data = get_next(feat_dim);
    std::vector<int> block_size = get_block_size();

    torch::Tensor ids = sample_data[0];
    torch::Tensor labels = sample_data[2];
    nvtxRangePop();

    
    //======================================== 混合数据结构读取 ========================================
    nvtx_msg = "ipc mixed data";
    nvtxRangePush(nvtx_msg.c_str());
    std::vector<vint> sparse_matrix_size = get_sparseMatrix_size();
    std::vector<torch::Tensor> sparse_matrix_data = get_sparseMatrix_data();

    vint bcsr_num_rows        = sparse_matrix_size[0];
    vint bcsr_num_cols        = sparse_matrix_size[1];
    vint bcsr_nnz             = sparse_matrix_size[2];
    vint bcsr_num_TCBlocks    = sparse_matrix_size[3];
    vint bcsr_num_rowWindows  = sparse_matrix_size[4];

    vint bcsc_num_rows        = sparse_matrix_size[5];
    vint bcsc_num_cols        = sparse_matrix_size[6];
    vint bcsc_nnz             = sparse_matrix_size[7];
    vint bcsc_num_TCBlocks    = sparse_matrix_size[8];
    vint bcsc_num_colWindows  = sparse_matrix_size[9];

    torch::Tensor bcsr_rowWindowOffset    = sparse_matrix_data[0];
    torch::Tensor bcsr_tcOffset           = sparse_matrix_data[1];
    torch::Tensor bcsr_sparseA2B          = sparse_matrix_data[2];
    torch::Tensor bcsr_tcLocalBit         = sparse_matrix_data[3];

    torch::Tensor bcsc_colWindowOffset               = sparse_matrix_data[4];
    torch::Tensor bcsc_tcOffset    = sparse_matrix_data[5];
    torch::Tensor bcsc_sparseA2C           = sparse_matrix_data[6];
    torch::Tensor bcsc_tcLocalBit          = sparse_matrix_data[7];
    nvtxRangePop();

    //======================================== feature 信息和数据读取 ========================================
    nvtx_msg = "ipc feature data";
    nvtxRangePush(nvtx_msg.c_str());
    torch::Tensor cache_search_map = get_memoryaccess_data()[0];
    std::vector<int64_t> feature_attribute = get_feature_attribute();
    std::vector<void*> feature_cache_ptr = get_feature_cache_ptr();

    int64_t cpu_float_feature_len   = feature_attribute[0];
    int64_t gpu_node_capacity       = feature_attribute[1];
    int64_t feature_dim             = feature_attribute[2];

    float* cpu_feature_cache_ptr = static_cast<float*>(feature_cache_ptr[0]);
    float** gpu_feature_cache_ptr = static_cast<float**>(feature_cache_ptr[1]);
    nvtxRangePop();

    //======================================== gcn first layer ========================================
    nvtx_msg = "AX";
    nvtxRangePush(nvtx_msg.c_str());
    torch::Tensor first_layer_out = forward_tensorcore_mixed(
        bcsr_num_rows, bcsr_num_cols, bcsr_nnz, bcsr_num_TCBlocks, bcsr_num_rowWindows,
        bcsr_rowWindowOffset, bcsr_tcOffset, bcsr_sparseA2B, bcsr_tcLocalBit,
        bcsc_num_rows, bcsc_num_cols, bcsc_nnz, bcsc_num_TCBlocks, bcsc_num_colWindows,
        bcsc_colWindowOffset, bcsc_tcOffset, bcsc_sparseA2C, bcsc_tcLocalBit,
        cpu_float_feature_len, gpu_node_capacity, feature_dim,
        ids, cpu_feature_cache_ptr, cache_search_map, gpu_feature_cache_ptr,
        device_id, denseC_first_layer
    );
    nvtxRangePop();

    #ifdef second
    // Reshape 1D -> 2D
    nvtx_msg = "reshape";
    nvtxRangePush(nvtx_msg.c_str());
    first_layer_out = first_layer_out.view({bcsr_num_rows, feature_dim});
    nvtxRangePop();

    // 初始化第一层权重矩阵
    nvtx_msg = "weight init";
    nvtxRangePush(nvtx_msg.c_str());
    //torch::Tensor weights_first_layer = torch::randn({feature_dim, hidden_dim}, torch::requires_grad()).to(device);
    torch::Tensor weights_first_layer = torch::empty({feature_dim, hidden_dim}, torch::requires_grad()).to(device);
    torch::nn::init::xavier_normal_(weights_first_layer);
    nvtxRangePop();

    // (AX)W
    nvtx_msg = "(AX)W";
    nvtxRangePush(nvtx_msg.c_str());
    first_layer_out = torch::mm(first_layer_out, weights_first_layer);
    nvtxRangePop();

    // 计算 ReLU 激活
    nvtx_msg = "relu";
    nvtxRangePush(nvtx_msg.c_str());
    first_layer_out = torch::relu(first_layer_out);
    nvtxRangePop();

    // 计算 Dropout（训练模式时生效）
    nvtx_msg = "dropout";
    nvtxRangePush(nvtx_msg.c_str());
    double dropout_prob = 0.5;
    torch::nn::Dropout dropout(dropout_prob);
    first_layer_out = dropout->forward(first_layer_out);
    nvtxRangePop();

    //======================================== gcn second layer ========================================
    // 3. Flatten 2D -> 1D，并确保数据是连续的
    nvtx_msg = "flatten";
    nvtxRangePush(nvtx_msg.c_str());
    torch::Tensor x = first_layer_out.flatten().contiguous().to(device);
    nvtxRangePop();

    torch::Tensor second_layer_out = forward_tensorcore(
        bcsr_num_rows, bcsr_num_cols, bcsr_nnz, bcsr_num_TCBlocks, bcsr_num_rowWindows,
        bcsr_rowWindowOffset, bcsr_tcOffset, bcsr_sparseA2B, bcsr_tcLocalBit,
        bcsc_num_rows, bcsc_num_cols, bcsc_nnz, bcsc_num_TCBlocks, bcsc_num_colWindows,
        bcsc_colWindowOffset, bcsc_tcOffset, bcsc_sparseA2C, bcsc_tcLocalBit,
        weight_vertical_axis_dim, x, device_id, denseC_second_layer
    );

    // Reshape 1D -> 2D
    second_layer_out = second_layer_out.view({bcsr_num_rows, hidden_dim});

    // 初始化第二层权重矩阵
    //torch::Tensor weights_second_layer = torch::randn({hidden_dim, num_classes}, torch::requires_grad()).to(device);
    torch::Tensor weights_second_layer = torch::empty({hidden_dim, num_classes}, torch::requires_grad()).to(device);
    torch::nn::init::xavier_normal_(weights_second_layer);

    // (AX)W
    second_layer_out = torch::mm(second_layer_out, weights_second_layer);

    labels = labels.to(torch::kLong);

    // 定义交叉熵损失函数
    // 计算损失
    // torch::nn::CrossEntropyLoss 本身不包含参数，不需要 to(device) 迁移，只需要确保输入的 second_layer_out 和 labels 在相同设备
    torch::nn::CrossEntropyLoss loss_fcn;
    torch::Tensor loss = loss_fcn(second_layer_out, labels);
    std::cout << "CrossEntropyLoss: " << loss.item<float>() << std::endl;
    #endif
    
    nvtx_msg = "Synchronize";
    nvtxRangePush(nvtx_msg.c_str());
    //freeMatC();
    Synchronize();
    nvtxRangePop();
}

void valid_one_step(int32_t feat_dim, int device_id, MAT_VAL_TYPE* denseC_first_layer, MAT_VAL_TYPE* denseC_second_layer) {
    torch::Device device(torch::kCUDA);

    
    //======================================== 获取数据 ========================================
    std::string nvtx_msg = "ipc original data";
    nvtxRangePush(nvtx_msg.c_str());
    std::vector<torch::Tensor> sample_data = get_next(feat_dim);
    std::vector<int> block_size = get_block_size();

    torch::Tensor ids = sample_data[0];
    torch::Tensor labels = sample_data[2];
    nvtxRangePop();

    
    //======================================== 混合数据结构读取 ========================================
    nvtx_msg = "ipc mixed data";
    nvtxRangePush(nvtx_msg.c_str());
    std::vector<vint> sparse_matrix_size = get_sparseMatrix_size();
    std::vector<torch::Tensor> sparse_matrix_data = get_sparseMatrix_data();

    vint bcsr_num_rows        = sparse_matrix_size[0];
    vint bcsr_num_cols        = sparse_matrix_size[1];
    vint bcsr_nnz             = sparse_matrix_size[2];
    vint bcsr_num_TCBlocks    = sparse_matrix_size[3];
    vint bcsr_num_rowWindows  = sparse_matrix_size[4];

    vint bcsc_num_rows        = sparse_matrix_size[5];
    vint bcsc_num_cols        = sparse_matrix_size[6];
    vint bcsc_nnz             = sparse_matrix_size[7];
    vint bcsc_num_TCBlocks    = sparse_matrix_size[8];
    vint bcsc_num_colWindows  = sparse_matrix_size[9];

    torch::Tensor bcsr_rowWindowOffset    = sparse_matrix_data[0];
    torch::Tensor bcsr_tcOffset           = sparse_matrix_data[1];
    torch::Tensor bcsr_sparseA2B          = sparse_matrix_data[2];
    torch::Tensor bcsr_tcLocalBit         = sparse_matrix_data[3];

    torch::Tensor bcsc_colWindowOffset               = sparse_matrix_data[4];
    torch::Tensor bcsc_tcOffset    = sparse_matrix_data[5];
    torch::Tensor bcsc_sparseA2C           = sparse_matrix_data[6];
    torch::Tensor bcsc_tcLocalBit          = sparse_matrix_data[7];
    nvtxRangePop();

    //======================================== feature 信息和数据读取 ========================================
    nvtx_msg = "ipc feature data";
    nvtxRangePush(nvtx_msg.c_str());
    torch::Tensor cache_search_map = get_memoryaccess_data()[0];
    std::vector<int64_t> feature_attribute = get_feature_attribute();
    std::vector<void*> feature_cache_ptr = get_feature_cache_ptr();

    int64_t cpu_float_feature_len   = feature_attribute[0];
    int64_t gpu_node_capacity       = feature_attribute[1];
    int64_t feature_dim             = feature_attribute[2];

    float* cpu_feature_cache_ptr = static_cast<float*>(feature_cache_ptr[0]);
    float** gpu_feature_cache_ptr = static_cast<float**>(feature_cache_ptr[1]);
    nvtxRangePop();

    //======================================== gcn first layer ========================================
    nvtx_msg = "first-layer-ax";
    nvtxRangePush(nvtx_msg.c_str());
    torch::Tensor first_layer_out = forward_tensorcore_mixed(
        bcsr_num_rows, bcsr_num_cols, bcsr_nnz, bcsr_num_TCBlocks, bcsr_num_rowWindows,
        bcsr_rowWindowOffset, bcsr_tcOffset, bcsr_sparseA2B, bcsr_tcLocalBit,
        bcsc_num_rows, bcsc_num_cols, bcsc_nnz, bcsc_num_TCBlocks, bcsc_num_colWindows,
        bcsc_colWindowOffset, bcsc_tcOffset, bcsc_sparseA2C, bcsc_tcLocalBit,
        cpu_float_feature_len, gpu_node_capacity, feature_dim,
        ids, cpu_feature_cache_ptr, cache_search_map, gpu_feature_cache_ptr,
        device_id, denseC_first_layer
    );
    nvtxRangePop();

    #ifdef second
    // Reshape 1D -> 2D
    first_layer_out = first_layer_out.view({bcsr_num_rows, feature_dim});

    // 初始化第一层权重矩阵
    //torch::Tensor weights_first_layer = torch::randn({feature_dim, hidden_dim}, torch::requires_grad()).to(device);
    torch::Tensor weights_first_layer = torch::empty({feature_dim, hidden_dim}, torch::requires_grad()).to(device);
    torch::nn::init::xavier_normal_(weights_first_layer);

    // (AX)W
    first_layer_out = torch::mm(first_layer_out, weights_first_layer);

    // 计算 ReLU 激活
    first_layer_out = torch::relu(first_layer_out);

    // 计算 Dropout（训练模式时生效）
    double dropout_prob = 0.5;
    torch::nn::Dropout dropout(dropout_prob);
    first_layer_out = dropout->forward(first_layer_out);

    //======================================== gcn second layer ========================================
    // 3. Flatten 2D -> 1D，并确保数据是连续的
    torch::Tensor x = first_layer_out.flatten().contiguous().to(device);

    torch::Tensor second_layer_out = forward_tensorcore(
        bcsr_num_rows, bcsr_num_cols, bcsr_nnz, bcsr_num_TCBlocks, bcsr_num_rowWindows,
        bcsr_rowWindowOffset, bcsr_tcOffset, bcsr_sparseA2B, bcsr_tcLocalBit,
        bcsc_num_rows, bcsc_num_cols, bcsc_nnz, bcsc_num_TCBlocks, bcsc_num_colWindows,
        bcsc_colWindowOffset, bcsc_tcOffset, bcsc_sparseA2C, bcsc_tcLocalBit,
        weight_vertical_axis_dim, x, device_id, denseC_second_layer
    );

    // Reshape 1D -> 2D
    second_layer_out = second_layer_out.view({bcsr_num_rows, hidden_dim});

    // 初始化第二层权重矩阵
    //torch::Tensor weights_second_layer = torch::randn({hidden_dim, num_classes}, torch::requires_grad()).to(device);
    torch::Tensor weights_second_layer = torch::empty({hidden_dim, num_classes}, torch::requires_grad()).to(device);
    torch::nn::init::xavier_normal_(weights_second_layer);

    // (AX)W
    second_layer_out = torch::mm(second_layer_out, weights_second_layer);

    labels = labels.to(torch::kLong);

    // 定义交叉熵损失函数
    // 计算损失
    // torch::nn::CrossEntropyLoss 本身不包含参数，不需要 to(device) 迁移，只需要确保输入的 second_layer_out 和 labels 在相同设备
    torch::nn::CrossEntropyLoss loss_fcn;
    torch::Tensor loss = loss_fcn(second_layer_out, labels);
    std::cout << "CrossEntropyLoss: " << loss.item<float>() << std::endl;
    #endif
    
    nvtx_msg = "Synchronize";
    nvtxRangePush(nvtx_msg.c_str());
    //freeMatC();
    Synchronize();
    nvtxRangePop();
}

void test_one_step(int32_t feat_dim, int device_id, MAT_VAL_TYPE* denseC_first_layer, MAT_VAL_TYPE* denseC_second_layer) {
    torch::Device device(torch::kCUDA);

    //======================================== 获取数据 ========================================
    std::string nvtx_msg = "ipc original data";
    nvtxRangePush(nvtx_msg.c_str());
    std::vector<torch::Tensor> sample_data = get_next(feat_dim);
    std::vector<int> block_size = get_block_size();

    torch::Tensor ids = sample_data[0];
    torch::Tensor labels = sample_data[2];
    nvtxRangePop();

    
    //======================================== 混合数据结构读取 ========================================
    nvtx_msg = "ipc mixed data";
    nvtxRangePush(nvtx_msg.c_str());
    std::vector<vint> sparse_matrix_size = get_sparseMatrix_size();
    std::vector<torch::Tensor> sparse_matrix_data = get_sparseMatrix_data();

    vint bcsr_num_rows        = sparse_matrix_size[0];
    vint bcsr_num_cols        = sparse_matrix_size[1];
    vint bcsr_nnz             = sparse_matrix_size[2];
    vint bcsr_num_TCBlocks    = sparse_matrix_size[3];
    vint bcsr_num_rowWindows  = sparse_matrix_size[4];

    vint bcsc_num_rows        = sparse_matrix_size[5];
    vint bcsc_num_cols        = sparse_matrix_size[6];
    vint bcsc_nnz             = sparse_matrix_size[7];
    vint bcsc_num_TCBlocks    = sparse_matrix_size[8];
    vint bcsc_num_colWindows  = sparse_matrix_size[9];

    torch::Tensor bcsr_rowWindowOffset    = sparse_matrix_data[0];
    torch::Tensor bcsr_tcOffset           = sparse_matrix_data[1];
    torch::Tensor bcsr_sparseA2B          = sparse_matrix_data[2];
    torch::Tensor bcsr_tcLocalBit         = sparse_matrix_data[3];

    torch::Tensor bcsc_colWindowOffset               = sparse_matrix_data[4];
    torch::Tensor bcsc_tcOffset    = sparse_matrix_data[5];
    torch::Tensor bcsc_sparseA2C           = sparse_matrix_data[6];
    torch::Tensor bcsc_tcLocalBit          = sparse_matrix_data[7];
    nvtxRangePop();

    //======================================== feature 信息和数据读取 ========================================
    nvtx_msg = "ipc feature data";
    nvtxRangePush(nvtx_msg.c_str());
    torch::Tensor cache_search_map = get_memoryaccess_data()[0];
    std::vector<int64_t> feature_attribute = get_feature_attribute();
    std::vector<void*> feature_cache_ptr = get_feature_cache_ptr();

    int64_t cpu_float_feature_len   = feature_attribute[0];
    int64_t gpu_node_capacity       = feature_attribute[1];
    int64_t feature_dim             = feature_attribute[2];

    float* cpu_feature_cache_ptr = static_cast<float*>(feature_cache_ptr[0]);
    float** gpu_feature_cache_ptr = static_cast<float**>(feature_cache_ptr[1]);
    nvtxRangePop();

    //======================================== gcn first layer ========================================
    nvtx_msg = "AX";
    nvtxRangePush(nvtx_msg.c_str());
    torch::Tensor first_layer_out = forward_tensorcore_mixed(
        bcsr_num_rows, bcsr_num_cols, bcsr_nnz, bcsr_num_TCBlocks, bcsr_num_rowWindows,
        bcsr_rowWindowOffset, bcsr_tcOffset, bcsr_sparseA2B, bcsr_tcLocalBit,
        bcsc_num_rows, bcsc_num_cols, bcsc_nnz, bcsc_num_TCBlocks, bcsc_num_colWindows,
        bcsc_colWindowOffset, bcsc_tcOffset, bcsc_sparseA2C, bcsc_tcLocalBit,
        cpu_float_feature_len, gpu_node_capacity, feature_dim,
        ids, cpu_feature_cache_ptr, cache_search_map, gpu_feature_cache_ptr,
        device_id, denseC_first_layer
    );
    nvtxRangePop();

    #ifdef second
    // Reshape 1D -> 2D
    nvtx_msg = "reshape";
    nvtxRangePush(nvtx_msg.c_str());
    first_layer_out = first_layer_out.view({bcsr_num_rows, feature_dim});
    nvtxRangePop();

    // 初始化第一层权重矩阵
    nvtx_msg = "weight init";
    nvtxRangePush(nvtx_msg.c_str());
    //torch::Tensor weights_first_layer = torch::randn({feature_dim, hidden_dim}, torch::requires_grad()).to(device);
    torch::Tensor weights_first_layer = torch::empty({feature_dim, hidden_dim}, torch::requires_grad()).to(device);
    torch::nn::init::xavier_normal_(weights_first_layer);
    nvtxRangePop();

    // (AX)W
    nvtx_msg = "(AX)W";
    nvtxRangePush(nvtx_msg.c_str());
    first_layer_out = torch::mm(first_layer_out, weights_first_layer);
    nvtxRangePop();

    // 计算 ReLU 激活
    nvtx_msg = "relu";
    nvtxRangePush(nvtx_msg.c_str());
    first_layer_out = torch::relu(first_layer_out);
    nvtxRangePop();


    // 计算 Dropout（训练模式时生效）
    nvtx_msg = "dropout";
    nvtxRangePush(nvtx_msg.c_str());
    double dropout_prob = 0.5;
    torch::nn::Dropout dropout(dropout_prob);
    first_layer_out = dropout->forward(first_layer_out);
    nvtxRangePop();

    //======================================== gcn second layer ========================================
    // 3. Flatten 2D -> 1D，并确保数据是连续的
    nvtx_msg = "flatten";
    nvtxRangePush(nvtx_msg.c_str());
    torch::Tensor x = first_layer_out.flatten().contiguous().to(device);
    nvtxRangePop();

    torch::Tensor second_layer_out = forward_tensorcore(
        bcsr_num_rows, bcsr_num_cols, bcsr_nnz, bcsr_num_TCBlocks, bcsr_num_rowWindows,
        bcsr_rowWindowOffset, bcsr_tcOffset, bcsr_sparseA2B, bcsr_tcLocalBit,
        bcsc_num_rows, bcsc_num_cols, bcsc_nnz, bcsc_num_TCBlocks, bcsc_num_colWindows,
        bcsc_colWindowOffset, bcsc_tcOffset, bcsc_sparseA2C, bcsc_tcLocalBit,
        weight_vertical_axis_dim, x, device_id, denseC_second_layer
    );

    // Reshape 1D -> 2D
    second_layer_out = second_layer_out.view({bcsr_num_rows, hidden_dim});

    // 初始化第二层权重矩阵
    //torch::Tensor weights_second_layer = torch::randn({hidden_dim, num_classes}, torch::requires_grad()).to(device);
    torch::Tensor weights_second_layer = torch::empty({hidden_dim, num_classes}, torch::requires_grad()).to(device);
    torch::nn::init::xavier_normal_(weights_second_layer);

    // (AX)W
    second_layer_out = torch::mm(second_layer_out, weights_second_layer);

    labels = labels.to(torch::kLong);

    // 定义交叉熵损失函数
    // 计算损失
    // torch::nn::CrossEntropyLoss 本身不包含参数，不需要 to(device) 迁移，只需要确保输入的 second_layer_out 和 labels 在相同设备
    torch::nn::CrossEntropyLoss loss_fcn;
    torch::Tensor loss = loss_fcn(second_layer_out, labels);
    std::cout << "CrossEntropyLoss: " << loss.item<float>() << std::endl;
    #endif
    
    nvtx_msg = "Synchronize";
    nvtxRangePush(nvtx_msg.c_str());
    //freeMatC();
    Synchronize();
    nvtxRangePop();
}

void RunnerLoop(int device_id, int gpu_number, int epoch, int feature_dim) {

    //bool is_waited = true;
    //while(is_waited);

    std::printf("[Info] GPU %d train\n", device_id);
    cudaSetDevice(device_id);

    std::string nvtx_msg = "initial space malloc and stream";
    nvtxRangePush(nvtx_msg.c_str());
    // 训练过程所需空间开辟
    spmm_cnic_initialize(device_id);

    int denseC_size = get_data_scale(device_id)[0];
    std::vector<void*> spaces = space_malloc(device_id, denseC_size);
    MAT_VAL_TYPE* denseC_first_layer = static_cast<MAT_VAL_TYPE*>(spaces[0]);
    MAT_VAL_TYPE* denseC_second_layer = static_cast<MAT_VAL_TYPE*>(spaces[1]);
    nvtxRangePop();

    // 等待采样进程 ready
    printf("device %d model instantiate end, waiting for sample ready...\n", device_id);
    //wait_sample_ready(device_id);
    printf("device %d train begin\n", device_id);

    InitializeIPC(gpu_number);

    std::vector<int32_t> steps = get_steps();

    for (int32_t i = 0; i < epoch; i++) {
        //if (device_id == 0) {
        //    std::printf("device-%d, epoch-%d, train_steps: %d, valid_steps: %d, test_steps: %d\n", device_id, i, steps[0], steps[1], steps[2]);
        //}

        //for (int32_t k = 0; k < steps[0]; k++) {
        for (int32_t k = 0; k < 50; k++) {
            nvtx_msg = "train one step(" + std::to_string(i) + "," + std::to_string(k) + ")";
            nvtxRangePush(nvtx_msg.c_str());
            train_one_step(feature_dim, device_id, denseC_first_layer, denseC_second_layer);
            nvtxRangePop();
        }
        for (int32_t k = 0; k < steps[1]; k++) {
            //train_one_step(feature_dim, device_id, denseC_first_layer, denseC_second_layer);
        }
    }

    for (int32_t k = 0; k < steps[2]; k++) {
        //train_one_step(feature_dim, device_id, denseC_first_layer, denseC_second_layer);
    }

}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: worker_main <gpu_id> <gpu_number> <epoch> <feature_dim>\n";
        return 1;
    }

    int gpu_id = std::stoi(argv[1]);
    int gpu_number = std::stoi(argv[2]);
    int epoch = std::stoi(argv[3]);
    int feature_dim = std::stoi(argv[4]);

    RunnerLoop(gpu_id, gpu_number, epoch, feature_dim);
    return 0;
}
