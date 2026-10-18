#include <cstdio>
#include <vector>
#include <thread>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <chrono>

#include <nvToolsExt.h>

#include <torch/csrc/distributed/c10d/Work.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>
#include <torch/csrc/autograd/profiler.h>
#include <torch/csrc/profiler/orchestration/observer.h>

#include "ipcservice.h"
#include "spmm_cnic.h"
#include "helper.h"
#include "models/gat.h"


void train_one_step(
    c10::intrusive_ptr<c10d::ProcessGroupNCCL> &pg,
    GAT &model,
    torch::optim::Adam &optimizer,
    torch::nn::CrossEntropyLoss &loss_fcn,
    int32_t feat_dim, 
    int32_t device_id, 
    int32_t gpu_number,
    MAT_VAL_TYPE* denseC_first_layer, MAT_VAL_TYPE* denseC_second_layer, int32_t denseC_size) {
    torch::Device device(torch::kCUDA, device_id);
    
    //======================================== 获取数据 ========================================
    std::string nvtx_msg = "ipc original data";
    nvtxRangePush(nvtx_msg.c_str());
    std::vector<torch::Tensor> sample_data = get_next(feat_dim);
    std::vector<int> block_size = get_block_size();

    torch::Tensor ids = sample_data[0];
    torch::Tensor features = sample_data[1];
    torch::Tensor labels = sample_data[2];
    nvtxRangePop();
    
    //======================================== 混合数据结构读取 ========================================
    nvtx_msg = "ipc mixed data";
    nvtxRangePush(nvtx_msg.c_str());
    std::vector<vint> sparse_matrix_size_vec = get_sparseMatrix_size();
    SparseMatrixSize sparse_matrix_size = {
        sparse_matrix_size_vec[0], sparse_matrix_size_vec[1], sparse_matrix_size_vec[2], sparse_matrix_size_vec[3], sparse_matrix_size_vec[4],
        sparse_matrix_size_vec[5], sparse_matrix_size_vec[6], sparse_matrix_size_vec[7], sparse_matrix_size_vec[8], sparse_matrix_size_vec[9]
    };

    std::vector<torch::Tensor> sparse_matrix_data_vec = get_sparseMatrix_data();
    SparseMatrixData sparse_matrix_data = {
        sparse_matrix_data_vec[0], sparse_matrix_data_vec[1], sparse_matrix_data_vec[2], sparse_matrix_data_vec[3],
        sparse_matrix_data_vec[4], sparse_matrix_data_vec[5], sparse_matrix_data_vec[6], sparse_matrix_data_vec[7]
    };
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

    //======================================== 模型前向计算 ========================================
    nvtx_msg = "train forward";
    nvtxRangePush(nvtx_msg.c_str());
    at::Tensor batch_pred = model(
        sparse_matrix_size,
        sparse_matrix_data,
        cpu_float_feature_len, gpu_node_capacity, feature_dim,
        features,
        ids, cpu_feature_cache_ptr, cache_search_map, gpu_feature_cache_ptr,
        device_id,
        denseC_size
    );
    batch_pred = batch_pred.index({torch::indexing::Slice(0, labels.size(0))}); // 截断，同 labels 对齐
    nvtxRangePop();

    #if 0
    //======================================== 计算 loss ========================================
    // TODO 需要确认一下是否需要 to(device)
    nvtx_msg = "compute loss";
    nvtxRangePush(nvtx_msg.c_str());
    labels = labels.to(torch::kLong);

    // 定义交叉熵损失函数
    // 计算损失
    torch::Tensor loss = loss_fcn(batch_pred, labels);
    //std::cout << "CrossEntropyLoss: " << loss.item<float>() << std::endl; // 需要同步一次，在需要评估精度时再开启
    nvtxRangePop();

    //======================================== 模型反向计算 ========================================
    nvtxRangePush("train backward");

    // 清空梯度 
    nvtxRangePush("zero_grad");
    optimizer.zero_grad();
    nvtxRangePop(); // zero_grad

    // 2. 反向传播 (等效于 loss.backward()) 仅计算导数
    nvtxRangePush("backward");
    loss.backward();
    nvtxRangePop(); // backward

    // 通信
    nvtxRangePush("allreduce gradients");

    #define split_allreduce
    #ifdef split_allreduce
    nvtxRangePush("allreduce");
    std::vector<c10::intrusive_ptr<c10d::Work>> works;
    for (auto& param : model->named_parameters()) {
        std::vector<torch::Tensor> tmp = {param.value().grad()};
        auto work = pg->allreduce(tmp);

        //ncclResult_t  ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
        //    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream);
    }
    nvtxRangePop(); // allreduce

    nvtxRangePush("grad normalization");
    for (auto& param : model->named_parameters()) {
        param.value().grad().div_(gpu_number);
    }
    nvtxRangePop(); // grad normalization
    #endif

    #ifdef merge_allreduce
    nvtxRangePush("allreduce");

    nvtxRangePush("data prepare");
    std::vector<torch::Tensor> grads;
    std::vector<torch::Tensor> flattens;
    std::vector<c10::IntArrayRef> original_shapes;
    for (auto& param : model->named_parameters()) {
        auto& grad = param.value().grad();
        original_shapes.push_back(grad.sizes());
        grads.push_back(grad);
        flattens.push_back(grad.flatten());
    }
    nvtxRangePop(); // data prepare

    // tensor 合并
    torch::Tensor merge_grads = torch::cat(flattens, 0);
    std::vector<torch::Tensor> input{merge_grads};

    // 通信
    c10::intrusive_ptr<c10d::Work> work = pg->allreduce(input);
    nvtxRangePop(); // allreduce

    nvtxRangePush("wait");
    work->wait();

    // tensor 切分
    std::vector<torch::Tensor> splits = merge_grads.split_with_sizes({
        flattens[0].numel(),  // 25600
        flattens[1].numel()   // 12032
    }, 0);

    nvtxRangePush("grad copy");
    // 恢复原始形状并写回
    grads[0].copy_(splits[0].view(original_shapes[0]));  // [25600] → [100,256]
    grads[1].copy_(splits[1].view(original_shapes[1]));  // [12032] → [256,47]
    nvtxRangePop(); // grad copy
    nvtxRangePop(); // wait

    nvtxRangePush("grad normalization");
    for (auto& param : model->named_parameters()) {
        param.value().grad().div_(gpu_number); // 原地除以 gpu_number
    }
    nvtxRangePop(); // grad normalization
    #endif

    nvtxRangePop(); // allreduce gradients
    nvtxRangePop(); // train backward

    // 3. 参数更新 (等效于 optimizer.step()) 包含 all_reduce 通信过程
    nvtxRangePush("parameters update");
    optimizer.step();
    nvtxRangePop();
    #endif

    nvtxRangePush("Synchronize");
    cudaDeviceSynchronize();
    //freeMatC();
    Synchronize();
    nvtxRangePop();
}

void valid_one_step(int32_t feat_dim, int device_id, MAT_VAL_TYPE* denseC_first_layer, MAT_VAL_TYPE* denseC_second_layer) {
}

void test_one_step(int32_t feat_dim, int device_id, MAT_VAL_TYPE* denseC_first_layer, MAT_VAL_TYPE* denseC_second_layer) {
}

void RunnerLoop(
    int32_t device_id, int32_t gpu_number, int32_t epoch,
    int32_t num_of_layers,
    std::vector<int32_t>& num_heads_per_layer,
    std::vector<int32_t>& num_features_per_layer) {

    std::printf("Running GNN Training on GPU %d\n", device_id);

    cudaSetDevice(device_id);
    torch::Device device(torch::kCUDA, device_id);

    //torch::autograd::profiler::enableProfiler(config);
    //torch::autograd::profiler::RecordProfile guard("trace.json");
    // 启动 Profiler（记录 CUDA 事件）
    torch::profiler::impl::ProfilerConfig cfg(
        torch::profiler::impl::ProfilerState::CUDA,
        true,   // report_input_shapes
        false,  // profile_memory
        true,   // with_stack
        true,   // with_flops
        false   // with_modules
    );
    //torch::autograd::profiler::RecordProfile guard("trace.json");
    torch::autograd::profiler::enableProfilerLegacy(cfg);

    // communition setting
    c10::intrusive_ptr<c10d::ProcessGroupNCCL> pg = setup(device_id, gpu_number);

    // 模型初始化
    std::string nvtx_msg;
    nvtxRangePush("model instantiate");
    // 实例化模型
    GAT model(
        num_of_layers, 
        num_heads_per_layer, 
        num_features_per_layer,
        false,       // 是否使用残差连接
        true,       // 是否添加 bias
        0.6         // dropout 概率
    );
    model->to(device);

    // torch::nn::CrossEntropyLoss 本身不包含参数，不需要 to(device) 迁移，只需要确保输入的 second_layer_out 和 labels 在相同设备
    torch::nn::CrossEntropyLoss loss_fcn;
    torch::optim::Adam optimizer(
        model->parameters(), // 获取模型的所有可训练参数
        torch::optim::AdamOptions(0.5) // 设置学习率
    );
    nvtxRangePop();

    nvtxRangePush("initial space malloc and stream");
    // 训练过程所需空间开辟
    spmm_cnic_initialize(device_id);

    int32_t denseC_size = get_data_scale(device_id)[0];
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
        //for (int32_t k = 0; k < steps[0]; k++) {
        for (int32_t k = 0; k < 50; k++) {
            nvtx_msg = "train one step(" + std::to_string(i) + "," + std::to_string(k) + ")";
            nvtxRangePush(nvtx_msg.c_str());
            train_one_step(pg, model, optimizer, loss_fcn, num_features_per_layer[0], device_id, gpu_number, denseC_first_layer, denseC_second_layer, denseC_size);
            nvtxRangePop();
        }
        for (int32_t k = 0; k < steps[1]; k++) {
            //train_one_step(feature_dim, device_id, denseC_first_layer, denseC_second_layer);
        }
    }

    for (int32_t k = 0; k < steps[2]; k++) {
        //train_one_step(feature_dim, device_id, denseC_first_layer, denseC_second_layer);
    }

    cleanup(pg);

    // 停止 Profiler 并获取事件
    auto event_lists = torch::autograd::profiler::disableProfilerLegacy();
    //std::ofstream trace_file("trace.json");
    //torch::autograd::profiler::writeProfilerEventsToStream(
    //    trace_file,
    //    // 需要将 event_lists 转换为 LegacyEvent* 的扁平列表
    //    get_flattened_events(event_lists)  // 需自定义此函数
    //);
    /*
    for (auto& events : event_lists) {
        for (auto& event : events) {
            std::cout << event.name() << ": " << event.cpuUs() << " us\n";
        }
    }
    */
}

int main(int argc, char** argv) {
    // if (argc != 8) {
    //     std::cerr << "Usage: worker_main <gpu_id> <num_classes> <feature_dim> <hidden_dim> <hops_num> <gpu_number> <epoch>\n";
    //     return 1;
    // }

    int32_t gpu_id = std::stoi(argv[1]);
    int32_t gpu_number = std::stoi(argv[2]);
    int32_t epoch = std::stoi(argv[3]);

    int32_t num_of_layers = std::stoi(argv[4]);

    std::vector<int32_t> num_heads_per_layer;
    for (int i = 0; i < num_of_layers; i++) {
        num_heads_per_layer.push_back(std::stoi(argv[4 + i]));
    }

    // 每一层都有输入和输出维度，因此 2 层有 3 个数据
    std::vector<int32_t> num_features_per_layer;
    for (int i = 0; i < num_of_layers + 1; i++) {
        num_features_per_layer.push_back(std::stoi(argv[4 + num_of_layers + i]));
    }

    try {
        RunnerLoop(gpu_id, gpu_number, epoch, num_of_layers, num_heads_per_layer, num_features_per_layer);
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        //return 1;
    }

    return 0;
}