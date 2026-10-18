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

#include "helper.h"

using ActivationFunc = std::function<at::Tensor(const at::Tensor&)>;

/**
 * 不同层中需要重复计算注意力吗？
 */

// -----------------------------
// Autograd Function (CNICSPMMFunction)
// -----------------------------
struct CNICSPMMFunction : public torch::autograd::Function<CNICSPMMFunction> {
    static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                                 int64_t bcsr_num_rows, int64_t bcsr_num_cols, int64_t bcsr_nnz,
                                 int64_t bcsr_num_TCBlocks, int64_t bcsr_num_rowWindows,
                                 at::Tensor bcsr_rowWindowOffset, at::Tensor bcsr_tcOffset, at::Tensor bcsr_sparseA2B, at::Tensor bcsr_tcLocalBit,
                                 int64_t bcsc_num_rows, int64_t bcsc_num_cols, int64_t bcsc_nnz,
                                 int64_t bcsc_num_TCBlocks, int64_t bcsc_num_colWindows,
                                 at::Tensor bcsc_colWindowOffset, at::Tensor bcsc_tcOffset, at::Tensor bcsc_sparseA2C, at::Tensor bcsc_tcLocalBit,
                                 int64_t cpu_float_feature_len, int64_t gpu_node_capacity, int64_t feature_dim,
                                 at::Tensor ids, float* cpu_feature_cache_ptr, at::Tensor cache_search_map, float**  gpu_feature_cache_ptr,
                                 at::Tensor x, int32_t device_id, at::Tensor denseC, int32_t denseC_size)
    {
        //first layer x shape[59522, 100]
        //second layer x shape[59522, 256]
        nvtxRangePush("AX");
        at::Tensor X_out;
        // 即使是二维 tensor 在转换为原始指针后可以正常连续访问，无需 flatten
        //at::Tensor x_flat = x.flatten().contiguous();
        X_out = forward_tensorcore(
            bcsr_num_rows, bcsr_num_cols, bcsr_nnz, bcsr_num_TCBlocks, bcsr_num_rowWindows,
            bcsr_rowWindowOffset, bcsr_tcOffset, bcsr_sparseA2B, bcsr_tcLocalBit,
            bcsc_num_rows, bcsc_num_cols, bcsc_nnz, bcsc_num_TCBlocks, bcsc_num_colWindows,
            bcsc_colWindowOffset, bcsc_tcOffset, bcsc_sparseA2C, bcsc_tcLocalBit,
            x, (int)device_id, denseC);
        nvtxRangePop();

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
        //at::Tensor denseC_tensor = torch::from_blob(
        //    denseC,                        // 数据指针
        //    {denseC_size},               // shape
        //    torch::kFloat32               // dtype
        //);
        //ctx->saved_data["denseC"] = denseC_tensor;
        ctx->saved_data["denseC"] = denseC;

        return X_out;
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
// Module: GATLayer，参考 GATLayer + GATLayerImp2/3 的实现
// -----------------------------
// 实现 BCSR 格式的注意力系数屏蔽
__global__ void maskScoreBCSR(
    float* connectivity_mask,
    vint bcsr_num_rows, vint bcsr_num_cols, vint bcsr_nnz, vint bcsr_num_TCBlocks, vint bcsr_num_rowWindows,
    vint* rowWindowOffset, vint* tcOffset, vint* sparseA2B, TCLOCAL_TYPE* tcLocalBit

) {
    int head_id = blockIdx.x;
    int window_id = threadIdx.x / 64;           // 两个 warp 作为一组负责一个 row_window
    int in_window_local_id = threadIdx.x % 64;   // window 内的局部线程 id
    
    // row_window 内 tcb 范围
    int tcb_start = rowWindowOffset[window_id];
    int tcb_end = rowWindowOffset[window_id + 1];

    // 迭代所在 window 内的 tc 块
    for (int i = tcb_start; i < tcb_end; i++) {
        // thread 解析 tcLocalBit 获取对应位置的数据
        TCLOCAL_TYPE present_local = tcLocalBit[i];

        // 如果线程对应到的位置是非零元
        if ((present_local >> in_window_local_id) & 0x1 == 1) {
            int global_row_indices = window_id * 8;     // row_window 全局起始行号
            int nnz_start = tcOffset[i];                // tcb 中非零元全局起始列号

            // 获取非零元的全局行列索引更新 connectivity_mask
            int row_idx = global_row_indices + (in_window_local_id / 8);    // 8 行一个 tc block
            int col_idx = sparseA2B[nnz_start + (in_window_local_id % 8)];  // 想拿到列号一定是访问 sparseA2B，nnz_start 作为基址, tcb 内的局部列号是局部偏移

            // connectivity_mask[head_id][row_idx][col_idx] = 0; 
            size_t index = (size_t)head_id * (bcsr_num_rows * bcsr_num_cols) + 
                           (size_t)row_idx * bcsr_num_cols + 
                           (size_t)col_idx;
            connectivity_mask[index] = 0; 
        }
    }
}
    
// 实现 BCSC 格式的注意力系数屏蔽
// TODO 需要 ipc 传入一个热数据的节点数量，作为冷数据列号的基址
__global__ void maskScoreBCSC(
    vint colData_baseAddr,
    float* connectivity_mask,
    vint bcsc_num_rows, vint bcsc_num_cols, vint bcsc_nnz, vint bcsc_num_TCBlocks, vint bcsc_num_colWindows,
    vint* colWindowOffset, vint* tcOffset, vint* sparseA2C, TCLOCAL_TYPE* tcLocalBit
) {
    int head_id = blockIdx.x;
    int window_id = threadIdx.x / 64;            // 两个 warp 作为一组负责一个 row_window
    int in_window_local_id = threadIdx.x % 64;   // window 内的局部线程 id
    
    // row_window 内 tcb 范围
    int tcb_start = colWindowOffset[window_id];
    int tcb_end = colWindowOffset[window_id + 1];

    // 迭代所在 window 内的 tc 块
    for (int i = tcb_start; i < tcb_end; i++) {
        // thread 解析 tcLocalBit 获取对应位置的数据
        TCLOCAL_TYPE present_local = tcLocalBit[i];

        // 如果线程对应到的位置是非零元
        if ((present_local >> in_window_local_id) & 0x1 == 1) {
            int global_col_indices = colData_baseAddr + window_id * 8;     // row_window 全局起始行号
            int nnz_start = tcOffset[i];                // tcb 中非零元全局起始列号

            // 获取非零元的全局行列索引更新 connectivity_mask
            int row_idx = sparseA2C[nnz_start + (in_window_local_id / 8)];  // 想拿到列号一定是访问 sparseA2B，nnz_start 作为基址, tcb 内的局部列号是局部偏移
            int col_idx = global_col_indices + (in_window_local_id % 8);    // 8 行一个 tc block

            //connectivity_mask[head_id][row_idx][col_idx] = 0; 
            size_t index = (size_t)head_id * (bcsc_num_rows * bcsc_num_cols) + 
                           (size_t)row_idx * bcsc_num_cols + 
                           (size_t)col_idx;
            connectivity_mask[index] = 0;
        }
    }
}

struct GATLayerImpl : public torch::nn::Module {
    // 现成模块
    ActivationFunc activation;
    torch::nn::Dropout dropout{nullptr};
    torch::nn::Linear linear_proj{nullptr};
    torch::nn::LeakyReLU leaky_relu{nullptr};
    torch::nn::Softmax softmax{nullptr};

    // 自定义可训练参数（模块封装了参数和计算逻辑，但是有些独特的需求需要单独定义参数）
    torch::Tensor scoring_fn_source;
    torch::Tensor scoring_fn_target;
    torch::Tensor bias;

    // 模型 layer 参数及选项
    int num_in_features;        // 不区分头的完整输入特征维度
    int num_out_features;       // 每个头持有的输出特征维度
    int num_of_heads;           // 头数
    bool add_skip_connection;
    bool add_bias;

    GATLayerImpl(int num_in_features, int num_out_features, int num_of_heads, ActivationFunc activation, double dropout_p, bool add_skip_connection, bool add_bias) : 
        num_in_features(num_in_features), num_out_features(num_out_features), num_of_heads(num_of_heads), activation(activation), add_skip_connection(add_skip_connection), add_bias(add_bias) {
        // 参数注册及初始化
        scoring_fn_source = register_parameter("scoring_fn_source", torch::empty({1, num_of_heads, num_out_features}));
        scoring_fn_target = register_parameter("scoring_fn_target", torch::empty({1, num_of_heads, num_out_features}));
        bias = register_parameter("bias", torch::empty(num_of_heads * num_out_features));

        torch::nn::init::xavier_normal_(scoring_fn_source);
        torch::nn::init::xavier_normal_(scoring_fn_target);
        torch::nn::init::zeros_(bias);

        // 模块注册
        dropout = register_module("dropout", 
            torch::nn::Dropout(
                torch::nn::DropoutOptions().p(dropout_p)
            )
        );
        linear_proj = register_module("linear_proj", torch::nn::Linear(num_in_features, num_of_heads * num_out_features));
        leaky_relu = register_module("leaky_relu", 
            torch::nn::LeakyReLU(
                torch::nn::LeakyReLUOptions().negative_slope(0.2)
            )
        );
        softmax = register_module("softmax", 
            torch::nn::Softmax(
                torch::nn::SoftmaxOptions(-1)
            )
        );
    }

    at::Tensor forward(const SparseMatrixSize &size, const SparseMatrixData &data,
                       int64_t cpu_float_feature_len, int64_t gpu_node_capacity, int64_t feature_dim,
                       at::Tensor ids, float* cpu_feature_cache_ptr, at::Tensor cache_search_map, float** gpu_feature_cache_ptr,
                       int32_t device_id, at::Tensor features, int32_t denseC_size)
    {
        // ====================== Step 1: Linear Projection + regularization（对特征数据进行处理，Whi）======================
        // 因为目前在采样过程中还是把特征数据都采集完全了，所以 features 数据可以直接使用
        at::Tensor in_nodes_features = dropout(features);
        // 线性变换 + view 负责拆分多头数据，(N, FIN) -> (N, NH * FOUT) -> (N, NH, FOUT)
        at::Tensor nodes_features_proj = linear_proj(in_nodes_features).view({-1, num_of_heads, num_out_features});
        nodes_features_proj = dropout(nodes_features_proj);

        // ====================== Step 2: Edge attention calculation（注意力系数计算）======================
        // (N, NH, FOUT) * (1, NH, FOUT) -> (N, NH, 1)（* represents element-wise (a.k.a. Hadamard) product)
        at::Tensor scores_source = torch::sum((nodes_features_proj * scoring_fn_source), /*dim=*/-1, /*keepdim=*/true);
        at::Tensor scores_target = torch::sum((nodes_features_proj * scoring_fn_target), /*dim=*/-1, /*keepdim=*/true);

        scores_source = scores_source.transpose(0, 1);      // (NH, N, 1)
        scores_target = scores_target.permute({1, 2, 0});   // (NH, 1, N)

        //scores_source size: [2, 59742, 1]
        //scores_target size: [2, 1, 59742]
        //scores_source size: [2, 62006, 1]
        //scores_target size: [2, 1, 62006]
        //std::cout << "scores_source size: " << scores_source.sizes() << std::endl;
        //std::cout << "scores_target size: " << scores_target.sizes() << std::endl;
        // TODO Impl3 在这里存在高效实现
        // 在加法过程中借助广播机制实现 (NH, N, 1) + (NH, 1, N) -> (NH, N, N) 
        at::Tensor all_scores = leaky_relu(scores_source + scores_target);

        // 这里的 connectivity_mask 有点难搞，因为既然已经有了自定义数据结构，就不能够再用传统的稠密矩阵表示了
        // 所以这里还需要借助于自定义存储结构中的信息来完成对注意力系数矩阵的修改，然后再改回到自定义存储结构的 value 字段中
        //at::Tensor all_attention_coefficients = softmax(all_scores + connectivity_mask);

        // TODO 需要将 softmax 后的计算结果修改到原始拓扑矩阵的 value 字段中
        // 朴素地来说，这里需要遍历每一个点的所有邻居，不过由于目前的数据组织方式已经变为了按 tensor core 的方式，所以只能按块遍历，修改 all_scores
        // all_scores 包含多头的数据，每个头拿到的都是完整的注意力系数，需要以同样的方式叠加拓扑数据, 目前先考虑朴素一点的实现，即每个 block 负责一个头的数据，问题在于头数比较少，sm 占用率较低，似乎用 warpgroup 更合适，64 个 thread 负责一个 window，迭代 window 内包含 64 个数据的 tc block
        // 直接修改 all_scores 的问题在于我们实际需要的是零元的位置，但是现有数据结构最多只能拿到 tcb 内的零元位置，并不是所有零元位置，所以暂时不行
        //maskScoreBCSR<<<NH, size.bcsr_num_rowWindows * 64>>>(all_scores, size, data);
        //maskScoreBCSC<<<NH, size.bcsc_num_colWindows * 64>>>(all_scores, size, data);
        //at::Tensor all_attention_coefficients = softmax(all_scores);
        
        #if 0
        at::Tensor connectivity_mask = torch::full(
            {size.bcsr_num_rows, size.bcsr_num_rows}, // 59840
            -std::numeric_limits<float>::infinity(),
            torch::kFloat
        );

        at::Tensor _out_nodes_features = torch::empty({num_of_heads, size.bcsr_num_rows, num_out_features});
        //try {

        // 非零元位置标记为 0
        // kernel 不能再传递 torch::tensor 了，编译有问题，只能传递为原始指针
        maskScoreBCSR<<<num_of_heads, size.bcsr_num_rowWindows * 64>>>(
            connectivity_mask.data_ptr<float>(),
            size.bcsr_num_rows, size.bcsr_num_cols, size.bcsr_nnz, size.bcsr_num_TCBlocks, size.bcsr_num_rowWindows,
            data.bcsr_rowWindowOffset.data_ptr<vint>(), data.bcsr_tcOffset.data_ptr<vint>(), data.bcsr_sparseA2B.data_ptr<vint>(), data.bcsr_tcLocalBit.data_ptr<TCLOCAL_TYPE>()
        );
        //TODO ipc 接口上添加热数据节点数量
        int colData_baseAddr = 0;
        maskScoreBCSC<<<num_of_heads, size.bcsc_num_colWindows * 64>>>(
            colData_baseAddr,
            connectivity_mask.data_ptr<float>(),
            size.bcsc_num_rows, size.bcsc_num_cols, size.bcsc_nnz, size.bcsc_num_TCBlocks, size.bcsc_num_colWindows,
            data.bcsc_colWindowOffset.data_ptr<vint>(), data.bcsc_tcOffset.data_ptr<vint>(), data.bcsc_sparseA2C.data_ptr<vint>(), data.bcsc_tcLocalBit.data_ptr<TCLOCAL_TYPE>()
        );

        at::Tensor all_attention_coefficients = softmax(all_scores + connectivity_mask);

        // ====================== Step 3: Neighborhood aggregation（特征聚合）======================
        // transpose(0, 1): (N, NH, FOUT) -> (NH, N, FOUT) 从以节点为中心转变为以注意力头为中心
        // (NH, N, N) * (NH, N, FOUT) -> (NH, N, FOUT)
        //at::Tensor out_nodes_features = torch::bmm(all_attention_coefficients, nodes_features_proj.transpose(0, 1))
        //at::Tensor out_nodes_features = torch::empty({num_of_heads, size.bcsr_num_rows, num_out_features});
        at::Tensor& out_nodes_features = _out_nodes_features;

        // 目前的实现性能次优的两点原因：1.头间计算并没有依赖关系，理论上可以并行计算的，但目前为串行；2.如果单纯是低一点问题，还能通过多流实现并行，但目前忽略了 value 的更新，如果加入后，因为 bcsr&&bcsc 的 value 只有一份空间，头数据的更新会强制其为串行，暂时还没有想到好方法解决这一点
        for (int i = 0; i < num_of_heads; i++) {
            // 下一步理论上是修改 bcsr 和 bcsc 的 value 字段，然后继续用 bcsr&&bcsc 完成特征聚合
            // 因为有多头数据，但是 value 就一个，所以在进行每一头计算之前都需要将对应头的注意力数据写入 value，这里暂时省略 value 的更新过程

            // 从 (NH, N, FOUT) 中选择第 i 个头的数据
            at::Tensor x = nodes_features_proj.select(0, i);
            at::Tensor denseC = out_nodes_features.select(0, i);

            CNICSPMMFunction::apply(
                size.bcsr_num_rows, size.bcsr_num_cols, size.bcsr_nnz, size.bcsr_num_TCBlocks, size.bcsr_num_rowWindows,
                data.bcsr_rowWindowOffset, data.bcsr_tcOffset, data.bcsr_sparseA2B, data.bcsr_tcLocalBit,
                size.bcsc_num_rows, size.bcsc_num_cols, size.bcsc_nnz, size.bcsc_num_TCBlocks, size.bcsc_num_colWindows,
                data.bcsc_colWindowOffset, data.bcsc_tcOffset, data.bcsc_sparseA2C, data.bcsc_tcLocalBit,
                cpu_float_feature_len, gpu_node_capacity, feature_dim,
                ids, cpu_feature_cache_ptr, cache_search_map, gpu_feature_cache_ptr,
                x, device_id, denseC, denseC_size
            );
        }

        // (NH, N, FOUT) -> (N, NH, FOUT)
        out_nodes_features = out_nodes_features.permute({1, 0, 2});

        // ====================== Step 4: Residual/skip connections, concat and bias ======================
        out_nodes_features = skip_concat_bias(all_attention_coefficients, in_nodes_features, out_nodes_features);
        #endif

        //} catch (const c10::Error& e) {
        //    std::cerr << "Error transferring connectivity_mask to GPU: " << e.what() << std::endl;
        //    throw;
        //}

        at::Tensor _out_nodes_features = torch::empty({num_of_heads, size.bcsr_num_rows, num_out_features});

        //return out_nodes_features;
        return _out_nodes_features;
    }

    // 残差连接→多头特征合并→偏置添加→激活函数
    at::Tensor skip_concat_bias(at::Tensor& all_attention_coefficients, at::Tensor& in_nodes_features, at::Tensor& out_nodes_features) {
        // 残差连接， 缓解深度模型的梯度消失问题
        if (add_skip_connection) {
            // in_node_features: (N, FIN) -> (N, 1, FIN)
            // (N, NH, FOUT) + (N, 1, FIN)，需要借助 broadcast
            out_nodes_features += in_nodes_features.unsqueeze(1);
        }

        // 多头特征合并，（N, NH, FOUT) -> (N, NH * FOUT);
        out_nodes_features = out_nodes_features.view({-1, num_of_heads * num_out_features});

        // 偏置添加
        if (add_bias) {
            out_nodes_features += bias;
        }

        // 激活
        if (activation != nullptr) {
            activation(out_nodes_features);
        }

        return out_nodes_features;
    }
};
TORCH_MODULE(GATLayer); // 生成名为 GATLayer 的智能指针，指向 GATLayerImpl 类，便于在模块间传递，可以从值拷贝转换为饮用拷贝

// -----------------------------
// Module: GAT
//      1. 层数
//      2. 每层的头数
//      3. 输入维度，输出维度
//
//  可训练参数：
//      1.linear projection matrix (denoted as "W" in the paper)
//      2.attention target/source (denoted as "a" in the paper)
//      3.bias (not mentioned in the paper but present in the official GAT repo)
// -----------------------------
struct GATImpl : public torch::nn::Module {
    std::vector<GATLayer> layers;

    // 输入示例数据：num_heads_per_layer：[8, 1]; num_features_per_layer:[1433, 8, 7]
    GATImpl(int num_of_layers,
            std::vector<int32_t>& num_heads_per_layer, 
            std::vector<int32_t>& num_features_per_layer, /*单头对应的 feature 维度*/
            bool add_skip_connection,
            bool add_bias,
            double dropout_p) {

        // 这是个 trick，主要原因在于希望仅用一个 num_features_per_layer 这样一个存储单头 feature 维度的数组同时存储输入和输出维度，但是第一层有些特殊
        num_heads_per_layer.insert(num_heads_per_layer.begin(), 1);

        
        for (int i = 0; i < num_of_layers; i++) {
            // 通过 register_module 让父子模块之间产生关联关系，方便 torch 在框架层面的统一管理
            layers.push_back(
                register_module(
                    "layer_" + std::to_string(i), 
                    GATLayer(
                        num_features_per_layer[i] * num_heads_per_layer[i],
                        num_features_per_layer[i + 1],
                        num_heads_per_layer[i + 1],
                        // 这里用 lambda 保证 elu，是因为 elu 的函数签名包含 4 个参数，但是实际我们一般只传递第一个，所以 activation 处的 function 也只写了一个函数，这里包装可以隐藏额外的参数
                        // 使用 ActivationFunc 的原因是三目运算符要求两个数据必须是相同类型或可以隐式地 转换为同一类型，但 lambda 和 nullptr 不兼容
                        (i < num_of_layers - 1) ? ActivationFunc([](const at::Tensor& input) -> at::Tensor { return torch::elu(input); }) : ActivationFunc(nullptr),    
                        dropout_p,
                        add_skip_connection,
                        add_bias
                    )
                )
            );
        }
    }

    at::Tensor forward(const SparseMatrixSize &size, const SparseMatrixData &data,
                       int64_t cpu_float_feature_len, int64_t gpu_node_capacity, int64_t feature_dim,
                       at::Tensor features,
                       at::Tensor ids, float* cpu_feature_cache_ptr, at::Tensor cache_search_map, float** gpu_feature_cache_ptr,
                       int32_t device_id,
                       int32_t denseC_size)
    {

        at::Tensor x = features;

        for (int i = 0; i < layers.size() - 1; i++) {
            std::string nvtx_msg = "forward first layer";
            nvtxRangePush(nvtx_msg.c_str());
            x = layers[i]->forward(size, data, cpu_float_feature_len, gpu_node_capacity, feature_dim,
                                  ids, cpu_feature_cache_ptr, cache_search_map, gpu_feature_cache_ptr,
                                  device_id, x, denseC_size);
            nvtxRangePop();

            // 层间操作放在这里还是 layer 内部？
        }

        return x;
    }
};
TORCH_MODULE(GAT);