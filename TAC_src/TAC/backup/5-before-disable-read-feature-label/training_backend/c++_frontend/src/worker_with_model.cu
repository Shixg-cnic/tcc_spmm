#include "ipcservice.h"
#include "spmm_cnic.h"
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

using namespace torch;

std::vector<torch::autograd::profiler::LegacyEvent*> get_flattened_events(
    const torch::autograd::profiler::thread_event_lists& event_lists) {
    std::vector<torch::autograd::profiler::LegacyEvent*> flattened;
    for (const auto& events : event_lists) {
        for (const auto& event : events) {
            // 注意：这里需要去掉const限定，因为writeProfilerEventsToStream需要非const指针
            flattened.push_back(const_cast<torch::autograd::profiler::LegacyEvent*>(&event));
        }
    }
    return flattened;
}

// 进程组管理的 API 由 torch 提供，将 ​​NCCL 的底层通信功能​​（如 ncclAllReduce）封装成 PyTorch 的分布式接口，运行时仍需要 NCCL 动态库支持
c10::intrusive_ptr<c10d::ProcessGroupNCCL> setup(int rank, int world_size) {
    // 设置主节点地址和端口
    std::string master_addr = "127.0.0.1";
    std::string master_port = "29500";
    
    // 创建 TCP store(基于 TCP 的分布式键值存储实现)
    auto store = c10::make_intrusive<c10d::TCPStore>(
        master_addr, 
        std::stoi(master_port),
        world_size,
        rank == 0,  // isServer
        std::chrono::seconds(30)
    );
    
    // 初始化 NCCL 进程组选项
    auto options = c10d::ProcessGroupNCCL::Options::create();
    options->global_ranks_in_group = std::vector<uint64_t>(world_size);
    for (int i = 0; i < world_size; ++i) {
        options->global_ranks_in_group[i] = i;
    }
    
    // 创建 NCCL 进程组
    c10::intrusive_ptr<c10d::ProcessGroupNCCL> pg = c10::make_intrusive<c10d::ProcessGroupNCCL>(
        store,
        rank,
        world_size,
        options
    );
    
    //c10d::initProcessGroup(pg);

    return pg;
}

void waitWork(
    c10::intrusive_ptr<c10d::ProcessGroupNCCL> pg,
    std::vector<c10::intrusive_ptr<c10d::Work>> works) {
    for (auto& work : works) {
      try {
        work->wait();
      } catch (const std::exception& ex) {
        std::cerr << "Exception received: " << ex.what() << std::endl;
        pg->abort();
      }
    }
}

inline void cleanup(c10::intrusive_ptr<c10d::ProcessGroupNCCL> pg) {
    // Difference between `abort()` and `shutdown()`:
    // 1. `abort()` will signal communicators to terminate all NCCL kernels
    // immediately.
    // 2. `shutdown()` will wait for all NCCL kernels to finish before destroying
    // communicators.

    // Destroy (shutdown) this backend -- normal exit.
    pg->shutdown();
}

// -----------------------------
// Helper types
// -----------------------------
struct SparseMatrixSize {
    int64_t bcsr_num_rows, bcsr_num_cols, bcsr_nnz, bcsr_num_TCBlocks, bcsr_num_rowWindows;
    int64_t bcsc_num_rows, bcsc_num_cols, bcsc_nnz, bcsc_num_TCBlocks, bcsc_num_colWindows;
};

struct SparseMatrixData {
    at::Tensor bcsr_rowWindowOffset, bcsr_tcOffset, bcsr_sparseA2B, bcsr_tcLocalBit;
    at::Tensor bcsc_colWindowOffset, bcsc_tcOffset, bcsc_sparseA2C, bcsc_tcLocalBit;
};

// -----------------------------
// Autograd Function (CNICSPMMFunction)
// -----------------------------
struct CNICSPMMFunction : public torch::autograd::Function<CNICSPMMFunction> {
    static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                                 bool first_layer,
                                 int64_t bcsr_num_rows, int64_t bcsr_num_cols, int64_t bcsr_nnz,
                                 int64_t bcsr_num_TCBlocks, int64_t bcsr_num_rowWindows,
                                 at::Tensor bcsr_rowWindowOffset, at::Tensor bcsr_tcOffset, at::Tensor bcsr_sparseA2B, at::Tensor bcsr_tcLocalBit,
                                 int64_t bcsc_num_rows, int64_t bcsc_num_cols, int64_t bcsc_nnz,
                                 int64_t bcsc_num_TCBlocks, int64_t bcsc_num_colWindows,
                                 at::Tensor bcsc_colWindowOffset, at::Tensor bcsc_tcOffset, at::Tensor bcsc_sparseA2C, at::Tensor bcsc_tcLocalBit,
                                 int64_t cpu_float_feature_len, int64_t gpu_node_capacity, int64_t feature_dim,
                                 at::Tensor ids, float* cpu_feature_cache_ptr, at::Tensor cache_search_map, float**  gpu_feature_cache_ptr,
                                 at::Tensor x, at::Tensor weights, int64_t weight_vertical_axis_dim, int32_t device_id, float* denseC, int32_t denseC_size)
    {
        //first layer x shape[59522, 100]
        //second layer x shape[59522, 256]
        nvtxRangePush("AX");
        at::Tensor X_out_flat;
        if (first_layer) {
            X_out_flat = forward_tensorcore_mixed(
                bcsr_num_rows, bcsr_num_cols, bcsr_nnz, bcsr_num_TCBlocks, bcsr_num_rowWindows,
                bcsr_rowWindowOffset, bcsr_tcOffset, bcsr_sparseA2B, bcsr_tcLocalBit,
                bcsc_num_rows, bcsc_num_cols, bcsc_nnz, bcsc_num_TCBlocks, bcsc_num_colWindows,
                bcsc_colWindowOffset, bcsc_tcOffset, bcsc_sparseA2C, bcsc_tcLocalBit,
                cpu_float_feature_len, gpu_node_capacity, feature_dim,
                ids, cpu_feature_cache_ptr, cache_search_map, gpu_feature_cache_ptr,
                x, (int)device_id, denseC);
        } else {
            // 即使是二维 tensor 在转换为原始指针后可以正常连续访问，无需 flatten
            //at::Tensor x_flat = x.flatten().contiguous();
            X_out_flat = forward_tensorcore(
                bcsr_num_rows, bcsr_num_cols, bcsr_nnz, bcsr_num_TCBlocks, bcsr_num_rowWindows,
                bcsr_rowWindowOffset, bcsr_tcOffset, bcsr_sparseA2B, bcsr_tcLocalBit,
                bcsc_num_rows, bcsc_num_cols, bcsc_nnz, bcsc_num_TCBlocks, bcsc_num_colWindows,
                bcsc_colWindowOffset, bcsc_tcOffset, bcsc_sparseA2C, bcsc_tcLocalBit,
                weight_vertical_axis_dim, x, (int)device_id, denseC);
        }
        nvtxRangePop();

        nvtxRangePush("reshape");
        at::Tensor X_out = X_out_flat.reshape({bcsr_num_rows, weight_vertical_axis_dim});
        nvtxRangePop();

        nvtxRangePush("(AX)W");
        at::Tensor X_gat = torch::mm(X_out, weights);
        nvtxRangePop();

        // 把参数保存到 context 中，以便在反向传播时使用
        ctx->save_for_backward({X_out, weights});

        ctx->saved_data["device_id"] = static_cast<int64_t>(device_id);
        ctx->saved_data["bcsr_num_rows"] = static_cast<int64_t>(bcsr_num_rows);
        ctx->saved_data["bcsr_num_cols"] = static_cast<int64_t>(bcsr_num_cols);
        ctx->saved_data["bcsr_nnz"] = static_cast<int64_t>(bcsr_nnz);
        ctx->saved_data["bcsr_num_TCBlocks"] = static_cast<int64_t>(bcsr_num_TCBlocks);
        ctx->saved_data["bcsr_num_rowWindows"] = static_cast<int64_t>(bcsr_num_rowWindows);
        ctx->saved_data["bcsc_num_rows"] = static_cast<int64_t>(bcsc_num_rows);
        ctx->saved_data["bcsc_num_cols"] = static_cast<int64_t>(bcsc_num_cols);
        ctx->saved_data["bcsc_nnz"] = static_cast<int64_t>(bcsc_nnz);
        ctx->saved_data["bcsc_num_TCBlocks"] = static_cast<int64_t>(bcsc_num_TCBlocks);
        ctx->saved_data["bcsc_num_colWindows"] = static_cast<int64_t>(bcsc_num_colWindows);

        ctx->saved_data["bcsr_rowWindowOffset"] = bcsr_rowWindowOffset;
        ctx->saved_data["bcsr_tcOffset"] = bcsr_tcOffset;
        ctx->saved_data["bcsr_sparseA2B"] = bcsr_sparseA2B;
        ctx->saved_data["bcsr_tcLocalBit"] = bcsr_tcLocalBit;
        ctx->saved_data["bcsc_colWindowOffset"] = bcsc_colWindowOffset;
        ctx->saved_data["bcsc_tcOffset"] = bcsc_tcOffset;
        ctx->saved_data["bcsc_sparseA2C"] = bcsc_sparseA2C;
        ctx->saved_data["bcsc_tcLocalBit"] = bcsc_tcLocalBit;

        // 无法保存原始指针 float*，需要构建 tensor 视图
        at::Tensor denseC_tensor = torch::from_blob(
            denseC,                        // 数据指针
            {denseC_size},               // shape
            torch::kFloat32               // dtype
        );
        ctx->saved_data["denseC"] = denseC_tensor;

        return X_gat;
    }

    static torch::autograd::variable_list backward(torch::autograd::AutogradContext *ctx,
                                                    torch::autograd::variable_list grad_outputs)
    {
        nvtxRangePush("prepare data");
        at::Tensor d_output = grad_outputs[0];

        auto saved = ctx->get_saved_variables();
        at::Tensor X_out = saved[0];
        at::Tensor weights = saved[1];

        int32_t device_id = static_cast<int32_t>(ctx->saved_data["device_id"].toInt());
        uint32_t bcsr_num_rows = static_cast<uint32_t>(ctx->saved_data["bcsr_num_rows"].toInt());
        uint32_t bcsr_num_cols = static_cast<uint32_t>(ctx->saved_data["bcsr_num_cols"].toInt());
        uint32_t bcsr_nnz = static_cast<uint32_t>(ctx->saved_data["bcsr_nnz"].toInt());
        uint32_t bcsr_num_TCBlocks = static_cast<uint32_t>(ctx->saved_data["bcsr_num_TCBlocks"].toInt());
        uint32_t bcsr_num_rowWindows = static_cast<uint32_t>(ctx->saved_data["bcsr_num_rowWindows"].toInt());
        uint32_t bcsc_num_rows = static_cast<uint32_t>(ctx->saved_data["bcsc_num_rows"].toInt());
        uint32_t bcsc_num_cols = static_cast<uint32_t>(ctx->saved_data["bcsc_num_cols"].toInt());
        uint32_t bcsc_nnz = static_cast<uint32_t>(ctx->saved_data["bcsc_nnz"].toInt());
        uint32_t bcsc_num_TCBlocks = static_cast<uint32_t>(ctx->saved_data["bcsc_num_TCBlocks"].toInt());
        uint32_t bcsc_num_colWindows = static_cast<uint32_t>(ctx->saved_data["bcsc_num_colWindows"].toInt());

        at::Tensor bcsr_rowWindowOffset = ctx->saved_data["bcsr_rowWindowOffset"].toTensor();
        at::Tensor bcsr_tcOffset = ctx->saved_data["bcsr_tcOffset"].toTensor();
        at::Tensor bcsr_sparseA2B = ctx->saved_data["bcsr_sparseA2B"].toTensor();
        at::Tensor bcsr_tcLocalBit = ctx->saved_data["bcsr_tcLocalBit"].toTensor();
        at::Tensor bcsc_colWindowOffset = ctx->saved_data["bcsc_colWindowOffset"].toTensor();
        at::Tensor bcsc_tcOffset = ctx->saved_data["bcsc_tcOffset"].toTensor();
        at::Tensor bcsc_sparseA2C = ctx->saved_data["bcsc_sparseA2C"].toTensor();
        at::Tensor bcsc_tcLocalBit = ctx->saved_data["bcsc_tcLocalBit"].toTensor();
        float* denseC = ctx->saved_data["denseC"].toTensor().data_ptr<float>();
        nvtxRangePop();

        nvtxRangePush("grad weight");
        at::Tensor grad_weights = at::mm(X_out.transpose(0,1), d_output);
        nvtxRangePop();

        nvtxRangePush("grad AX");
        at::Tensor grad_AX = at::mm(d_output, weights.transpose(0,1));
        nvtxRangePop();

        nvtxRangePush("grad X");
        at::Tensor grad_AX_flat = grad_AX.flatten().contiguous();
        at::Tensor grad_X_flat = forward_tensorcore_backward(
            bcsr_num_rows, bcsr_num_cols, bcsr_nnz, bcsr_num_TCBlocks, bcsr_num_rowWindows,
            bcsr_rowWindowOffset, bcsr_tcOffset, bcsr_sparseA2B, bcsr_tcLocalBit,
            bcsc_num_rows, bcsc_num_cols, bcsc_nnz, bcsc_num_TCBlocks, bcsc_num_colWindows,
            bcsc_colWindowOffset, bcsc_tcOffset, bcsc_sparseA2C, bcsc_tcLocalBit,
            grad_AX.sizes()[1], grad_AX_flat, (int)device_id, denseC);
        nvtxRangePop();

        nvtxRangePush("grad X reshape");
        at::Tensor grad_X = grad_X_flat.reshape({bcsr_num_rows, grad_AX.sizes()[1]});
        nvtxRangePop();

        nvtxRangePush("prepare return data");
        // Build return list matching forward argument list length
        torch::autograd::variable_list grads;
        // first_layer
        grads.push_back(torch::Tensor());
        // bcsr scalars
        for (int i=0;i<5;i++) grads.push_back(torch::Tensor());
        // bcsr tensor args
        grads.push_back(torch::Tensor());
        grads.push_back(torch::Tensor());
        grads.push_back(torch::Tensor());
        grads.push_back(torch::Tensor());
        // bcsc scalars
        for (int i=0;i<5;i++) grads.push_back(torch::Tensor());
        // bcsc tensor args
        grads.push_back(torch::Tensor());
        grads.push_back(torch::Tensor());
        grads.push_back(torch::Tensor());
        grads.push_back(torch::Tensor());
        // cpu_float_feature_len, gpu_node_capacity, feature_dim
        grads.push_back(torch::Tensor());
        grads.push_back(torch::Tensor());
        grads.push_back(torch::Tensor());
        // ids, cpu_feature_cache_ptr, cache_search_map, gpu_feature_cache_ptr
        grads.push_back(torch::Tensor());
        grads.push_back(torch::Tensor());
        grads.push_back(torch::Tensor());
        grads.push_back(torch::Tensor());
        // x
        grads.push_back(grad_X);
        // weights
        grads.push_back(grad_weights);
        // weight_vertical_axis_dim, device_id, denseC, denseC_size
        grads.push_back(torch::Tensor());
        grads.push_back(torch::Tensor());
        grads.push_back(torch::Tensor());
        grads.push_back(torch::Tensor());
        nvtxRangePop();

        return grads;
    }
};

// -----------------------------
// Module: GCNConv
// -----------------------------
struct GCNConvImpl : public torch::nn::Module {
    bool first_layer;
    torch::Tensor weights;
    int64_t weight_vertical_axis_dim;

    GCNConvImpl(int input_dim, int output_dim, bool first_layer_ = false) : first_layer(first_layer_) {
        // 权重初始化
        weights = register_parameter("weights", torch::empty({input_dim, output_dim}));
        torch::nn::init::xavier_normal_(weights);
        weight_vertical_axis_dim = input_dim;
    }

    at::Tensor forward(const SparseMatrixSize &size, const SparseMatrixData &data,
                       int64_t cpu_float_feature_len, int64_t gpu_node_capacity, int64_t feature_dim,
                       at::Tensor ids, float* cpu_feature_cache_ptr, at::Tensor cache_search_map, float** gpu_feature_cache_ptr,
                       int32_t device_id, at::Tensor x, float* denseC, int32_t denseC_size)
    {
        return CNICSPMMFunction::apply(
            first_layer,
            size.bcsr_num_rows, size.bcsr_num_cols, size.bcsr_nnz, size.bcsr_num_TCBlocks, size.bcsr_num_rowWindows,
            data.bcsr_rowWindowOffset, data.bcsr_tcOffset, data.bcsr_sparseA2B, data.bcsr_tcLocalBit,
            size.bcsc_num_rows, size.bcsc_num_cols, size.bcsc_nnz, size.bcsc_num_TCBlocks, size.bcsc_num_colWindows,
            data.bcsc_colWindowOffset, data.bcsc_tcOffset, data.bcsc_sparseA2C, data.bcsc_tcLocalBit,
            cpu_float_feature_len, gpu_node_capacity, feature_dim,
            ids, cpu_feature_cache_ptr, cache_search_map, gpu_feature_cache_ptr,
            x, weights, weight_vertical_axis_dim, device_id, denseC, denseC_size
        );
    }
};
TORCH_MODULE(GCNConv);

// -----------------------------
// Module: GCN
// -----------------------------
struct GCNImpl : public torch::nn::Module {
    int64_t num_classes;
    GCNConv conv1{nullptr};
    GCNConv conv_final{nullptr};
    torch::nn::Dropout dropout{nullptr};
    std::function<at::Tensor(const at::Tensor&)> activation;

    GCNImpl(int64_t num_features, int64_t hidden_dim, int64_t num_classes_, int num_layers,
            std::function<at::Tensor(const at::Tensor&)> activation_, double dropout_p) : num_classes(num_classes_), activation(activation_) {

        conv1 = register_module("conv1", GCNConv(num_features, hidden_dim, true));
        conv_final = register_module("conv_final", GCNConv(hidden_dim, num_classes_));
        // dropout 仅在训练时启用，因此涉及到状态切换，需要注册为模块便于控制；relu 的使用规则在训练和推理中没有区别，无需切换状态
        dropout = register_module("dropout", torch::nn::Dropout(torch::nn::DropoutOptions().p(dropout_p)));
    }

    at::Tensor forward(const SparseMatrixSize &size, const SparseMatrixData &data,
                       int64_t cpu_float_feature_len, int64_t gpu_node_capacity, int64_t feature_dim,
                       at::Tensor features,
                       at::Tensor ids, float* cpu_feature_cache_ptr, at::Tensor cache_search_map, float** gpu_feature_cache_ptr,
                       int32_t device_id,
                       float* denseC_first_layer,
                       float* denseC_second_layer,
                       int32_t denseC_size)
    {
        at::Tensor x = features;
        std::string nvtx_msg = "forward first layer";
        nvtxRangePush(nvtx_msg.c_str());
        x = conv1->forward(size, data, cpu_float_feature_len, gpu_node_capacity, feature_dim,
                            ids, cpu_feature_cache_ptr, cache_search_map, gpu_feature_cache_ptr,
                            device_id, x, denseC_first_layer, denseC_size);
        nvtxRangePop();

        nvtx_msg = "relu";
        nvtxRangePush(nvtx_msg.c_str());
        x = activation(x);
        nvtxRangePop();

        nvtx_msg = "dropout";
        nvtxRangePush(nvtx_msg.c_str());
        x = dropout(x);
        nvtxRangePop();

        nvtx_msg = "forward second layer";
        nvtxRangePush(nvtx_msg.c_str());
        x = conv_final->forward(size, data, cpu_float_feature_len, gpu_node_capacity, feature_dim,
                                 ids, cpu_feature_cache_ptr, cache_search_map, gpu_feature_cache_ptr,
                                 device_id, x, denseC_second_layer, denseC_size);
        nvtxRangePop();

        return x;
    }
};
TORCH_MODULE(GCN);

void train_one_step(
    c10::intrusive_ptr<c10d::ProcessGroupNCCL> &pg,
    GCN &model,
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
        denseC_first_layer,
        denseC_second_layer,
        denseC_size
    );
    batch_pred = batch_pred.index({torch::indexing::Slice(0, labels.size(0))}); // 截断，同 labels 对齐
    nvtxRangePop();

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
    int32_t device_id, 
    int32_t num_classes, int32_t feature_dim, int32_t hidden_dim, 
    int32_t hops_num, 
    int32_t gpu_number, 
    int32_t epoch) {

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

    // 初始化第一层权重矩阵
    std::string nvtx_msg;
    nvtxRangePush("model instantiate");
    // TODO 添加 argument 类传递参数
    // 实例化模型
    GCN model(
        feature_dim, 
        hidden_dim, 
        num_classes,
        2,           // 两层 GCN
        torch::relu, // 激活函数
        0.5         // dropout 概率
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
        for (int32_t k = 0; k < 25; k++) {
            nvtx_msg = "train one step(" + std::to_string(i) + "," + std::to_string(k) + ")";
            nvtxRangePush(nvtx_msg.c_str());
            train_one_step(pg, model, optimizer, loss_fcn, feature_dim, device_id, gpu_number, denseC_first_layer, denseC_second_layer, denseC_size);
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
    if (argc != 8) {
        std::cerr << "Usage: worker_main <gpu_id> <num_classes> <feature_dim> <hidden_dim> <hops_num> <gpu_number> <epoch>\n";
        return 1;
    }

    int32_t gpu_id = std::stoi(argv[1]);
    int32_t num_classes = std::stoi(argv[2]);
    int32_t feature_dim = std::stoi(argv[3]);
    int32_t hidden_dim = std::stoi(argv[4]);
    int32_t hops_num = std::stoi(argv[5]);
    int32_t gpu_number = std::stoi(argv[6]);
    int32_t epoch = std::stoi(argv[7]);

    try {
        RunnerLoop(gpu_id, num_classes, feature_dim, hidden_dim, hops_num, gpu_number, epoch);
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        //return 1;
    }

    return 0;
}
