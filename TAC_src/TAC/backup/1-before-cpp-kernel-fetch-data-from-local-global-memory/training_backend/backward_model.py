#!/usr/bin/env python3
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import sys
import tempfile
import argparse
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn.functional as F

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as Func

import ipc_service
import spmm_cnic

import time
import numpy as np
import torchmetrics
torch.set_printoptions(threshold=np.inf)

import csv
from time import perf_counter

# 文档见: https://nvtx.readthedocs.io/en/latest/annotate.html
import nvtx

#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from collections import namedtuple

SparseMatrixSize = namedtuple('SparseMatrixSize', [
    'bcsr_num_rows', 'bcsr_num_cols', 'bcsr_nnz', 'bcsr_num_TCBlocks', 'bcsr_num_rowWindows',
    'bcsc_num_rows', 'bcsc_num_cols', 'bcsc_nnz', 'bcsc_num_TCBlocks', 'bcsc_num_colWindows'
])

SparseMatrixData = namedtuple('SparseMatrixData', [
    'bcsr_rowWindowOffset', 'bcsr_tcOffset', 'bcsr_sparseA2B', 'bcsr_tcLocalBit', 
    'bcsc_colWindowOffset', 'bcsc_tcOffset', 'bcsc_sparseA2C', 'bcsc_tcLocalBit'
])

class CNICSPMMFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        first_layer,
        sparse_matrix_size,
        sparse_matrix_data, 
        cpu_float_feature_len, gpu_node_capacity, feature_dim,
        ids, cpu_feature_cache_ptr, cache_search_map, gpu_feature_cache_ptr,
        x, weights, weight_vertical_axis_dim, device_id, denseC):

        # bcsr
        bcsr_num_rows        = sparse_matrix_size.bcsr_num_rows
        bcsr_num_cols        = sparse_matrix_size.bcsr_num_cols
        bcsr_nnz             = sparse_matrix_size.bcsr_nnz
        bcsr_num_TCBlocks    = sparse_matrix_size.bcsr_num_TCBlocks
        bcsr_num_rowWindows  = sparse_matrix_size.bcsr_num_rowWindows
        bcsr_rowWindowOffset = sparse_matrix_data.bcsr_rowWindowOffset
        bcsr_tcOffset        = sparse_matrix_data.bcsr_tcOffset
        bcsr_sparseA2B       = sparse_matrix_data.bcsr_sparseA2B
        bcsr_tcLocalBit      = sparse_matrix_data.bcsr_tcLocalBit
        # bcsc
        bcsc_num_rows        = sparse_matrix_size.bcsc_num_rows
        bcsc_num_cols        = sparse_matrix_size.bcsc_num_cols
        bcsc_nnz             = sparse_matrix_size.bcsc_nnz
        bcsc_num_TCBlocks    = sparse_matrix_size.bcsc_num_TCBlocks
        bcsc_num_colWindows  = sparse_matrix_size.bcsc_num_colWindows
        bcsc_colWindowOffset = sparse_matrix_data.bcsc_colWindowOffset
        bcsc_tcOffset        = sparse_matrix_data.bcsc_tcOffset
        bcsc_sparseA2C       = sparse_matrix_data.bcsc_sparseA2C
        bcsc_tcLocalBit      = sparse_matrix_data.bcsc_tcLocalBit

        # 构建节点编号映射表（第一层采用实际节点编号，其他层采用顺序编号，用于 SpMM 过程中 MatB 的访存）
        if first_layer == True:
            rng = nvtx.start_range(message=("AX"), color="green") 
            X_out = spmm_cnic.forward_tensorcore_mixed(
                bcsr_num_rows, bcsr_num_cols, bcsr_nnz, bcsr_num_TCBlocks, bcsr_num_rowWindows,
                bcsr_rowWindowOffset, bcsr_tcOffset, bcsr_sparseA2B, bcsr_tcLocalBit,
                bcsc_num_rows, bcsc_num_cols, bcsc_nnz, bcsc_num_TCBlocks, bcsc_num_colWindows,
                bcsc_colWindowOffset, bcsc_tcOffset, bcsc_sparseA2C, bcsc_tcLocalBit,
                cpu_float_feature_len, gpu_node_capacity, feature_dim,
                ids, cpu_feature_cache_ptr, cache_search_map, gpu_feature_cache_ptr,
                device_id, denseC)
            nvtx.end_range(rng) 
        else:
            x = x.flatten().contiguous()
            rng = nvtx.start_range(message=("AX"), color="green") 
            X_out = spmm_cnic.forward_tensorcore(
                bcsr_num_rows, bcsr_num_cols, bcsr_nnz, bcsr_num_TCBlocks, bcsr_num_rowWindows,
                bcsr_rowWindowOffset, bcsr_tcOffset, bcsr_sparseA2B, bcsr_tcLocalBit,
                bcsc_num_rows, bcsc_num_cols, bcsc_nnz, bcsc_num_TCBlocks, bcsc_num_colWindows,
                bcsc_colWindowOffset, bcsc_tcOffset, bcsc_sparseA2C, bcsc_tcLocalBit,
                weight_vertical_axis_dim, x, device_id, denseC)
            nvtx.end_range(rng) 

        # 1D -> 2D matrix
        X_out = X_out.reshape(bcsr_num_rows, weight_vertical_axis_dim)
        rng = nvtx.start_range(message=("(AX)W"), color="green") 

        #print(f"X_out: shape={X_out.shape}, dtype={X_out.dtype}, device={X_out.device}", flush=True)
        #print(f"weights: shape={weights.shape}, dtype={weights.dtype}, device={weights.device}", flush=True)
        #assert X_out.ndim == 2 and weights.ndim == 2, "Expected 2D tensors"
        #assert X_out.shape[1] == weights.shape[0], f"Shape mismatch: {X_out.shape} vs {weights.shape}"
        #assert X_out.device == weights.device, f"Device mismatch: {X_out.device} vs {weights.device}"
        #assert X_out.dtype == weights.dtype, f"Dtype mismatch: {X_out.dtype} vs {weights.dtype}"
        #assert X_out.dtype in [torch.float32, torch.float64, torch.float16, torch.bfloat16], f"Unsupported dtype: {X_out.dtype}"

        X_gat = torch.mm(X_out, weights)
        nvtx.end_range(rng) 

        #if first_layer == False:
        #    while True:
        #        time.sleep(1)

        # 现在不需要了
        # indices = torch.stack([row_indices, col_indices]) # (2, N) 的张量
        # values = torch.ones(indices.shape[1]).to(device)  # 构造出来
        # A_topo = torch.sparse_coo_tensor(indices, values, size=(bcsr_num_rows, bcsr_num_cols))

        # 把参数保存到 context 中，以便在反向传播时使用
        ctx.device_id = device_id
        ctx.save_for_backward(X_out, weights)
        ctx.topo_data = (bcsr_num_rows, bcsr_num_cols, bcsr_nnz, bcsr_num_TCBlocks, bcsr_num_rowWindows,
                        bcsr_rowWindowOffset, bcsr_tcOffset, bcsr_sparseA2B, bcsr_tcLocalBit,
                        bcsc_num_rows, bcsc_num_cols, bcsc_nnz, bcsc_num_TCBlocks, bcsc_num_colWindows,
                        bcsc_colWindowOffset, bcsc_tcOffset, bcsc_sparseA2C, bcsc_tcLocalBit, denseC)
        # ctx.weight_dim = weight_vertical_axis_dim

        return X_gat

    @staticmethod
    def backward(ctx, d_output):
        # 1. 获取 context 保存的中间变量
        device_id = ctx.device_id
        X_out, weights = ctx.saved_tensors # 返回的是一个 tuple，所以要用逗号解包
        bcsr_num_rows, bcsr_num_cols, bcsr_nnz, bcsr_num_TCBlocks, bcsr_num_rowWindows, \
        bcsr_rowWindowOffset, bcsr_tcOffset, bcsr_sparseA2B, bcsr_tcLocalBit,  \
        bcsc_num_rows, bcsc_num_cols, bcsc_nnz, bcsc_num_TCBlocks, bcsc_num_colWindows, \
        bcsc_colWindowOffset, bcsc_tcOffset, bcsc_sparseA2C, bcsc_tcLocalBit, denseC = ctx.topo_data
        # weight_vertical_axis_dim = ctx.weight_dim

        rng = nvtx.start_range(message=("grad weight"), color="green") 
        grad_weights = torch.mm(X_out.T, d_output)
        nvtx.end_range(rng) 

        # 2. 先反向传播过 dense matmul
        rng = nvtx.start_range(message=("grad AX"), color="green") 
        grad_AX = torch.mm(d_output, weights.T) 
        nvtx.end_range(rng) 

        rng = nvtx.start_range(message=("grad X"), color="green") 
        grad_X = spmm_cnic.forward_tensorcore(
                bcsr_num_rows, bcsr_num_cols, bcsr_nnz, bcsr_num_TCBlocks, bcsr_num_rowWindows,
                bcsr_rowWindowOffset, bcsr_tcOffset, bcsr_sparseA2B, bcsr_tcLocalBit,
                bcsc_num_rows, bcsc_num_cols, bcsc_nnz, bcsc_num_TCBlocks, bcsc_num_colWindows,
                bcsc_colWindowOffset, bcsc_tcOffset, bcsc_sparseA2C, bcsc_tcLocalBit,
                grad_AX.shape[1], grad_AX.flatten().contiguous(), device_id, denseC)
        nvtx.end_range(rng) 

        rng = nvtx.start_range(message=("grad X reshape"), color="green") 
        grad_X = grad_X.reshape(bcsr_num_rows, grad_AX.shape[1])
        nvtx.end_range(rng) 

        # 4. 返回梯度(返回值必须和 forward 的参数数量/位置一一对应, 非张量（标量、int、bool）或者不需要梯度的参数 → 返回 None)
        return (
            None,
            None, None, 
            None, None, None,
            None, None, None, None,
            grad_X,
            grad_weights,  # weights 的梯度
            None, None, None
        )


class GCNConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, first_layer=False):
        super(GCNConv, self).__init__()
        self.first_layer = first_layer
        #print("input_dim: {}, output_dim: {}".format(input_dim, output_dim), flush=True)
        #self.weights = torch.nn.Parameter(torch.randn(input_dim, output_dim))
        rng = nvtx.start_range(message=("weights init"), color="green") 
        self.weights = torch.nn.Parameter(torch.empty(input_dim, output_dim))
        torch.nn.init.xavier_uniform_(self.weights)
        nvtx.end_range(rng) 
        self.weight_vertical_axis_dim = input_dim

    def forward(
        self,
        sparse_matrix_size,
        sparse_matrix_data, 
        cpu_float_feature_len, gpu_node_capacity, feature_dim,
        ids, cpu_feature_cache_ptr, cache_search_map, gpu_feature_cache_ptr,
        device_id, x, denseC):

        return CNICSPMMFunction.apply(
            self.first_layer,
            sparse_matrix_size,
            sparse_matrix_data, 
            cpu_float_feature_len, gpu_node_capacity, feature_dim,
            ids, cpu_feature_cache_ptr, cache_search_map, gpu_feature_cache_ptr,
            x, self.weights, self.weight_vertical_axis_dim, device_id, denseC)

class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, num_layers, activation, dropout):
        super(GCN, self).__init__()
        self.num_classes = num_classes

        # 第一层
        # 调用具体 layer 的 init 方法
        self.conv1 = GCNConv(num_features, hidden_dim, first_layer=True)
        
        # 最后一层
        self.conv_final = GCNConv(hidden_dim, num_classes)
        
        self.dropout = nn.Dropout(p=dropout)
        self.relu = activation

    def forward(
        self,
        sparse_matrix_size,
        sparse_matrix_data, 
        cpu_float_feature_len, gpu_node_capacity, feature_dim,
        ids, cpu_feature_cache_ptr, cache_search_map, gpu_feature_cache_ptr,
        device_id,
        denseC_first_layer,
        denseC_second_layer
    ):
        # 第一层
        x = torch.empty(0)
        rng = nvtx.start_range(message=("forward first layer"), color="green")
        x = self.conv1(
                sparse_matrix_size,
                sparse_matrix_data, 
                cpu_float_feature_len, gpu_node_capacity, feature_dim,
                ids, cpu_feature_cache_ptr, cache_search_map, gpu_feature_cache_ptr, 
                device_id, x, denseC_first_layer)
        nvtx.end_range(rng)

        rng = nvtx.start_range(message=("relu"), color="green")
        x = self.relu(x)
        nvtx.end_range(rng)

        rng = nvtx.start_range(message=("dropout"), color="green")
        x = self.dropout(x)
        nvtx.end_range(rng)
        
        # 最后一层
        rng = nvtx.start_range(message=("forward second layer"), color="green")
        x = self.conv_final(
                sparse_matrix_size,
                sparse_matrix_data, 
                cpu_float_feature_len, gpu_node_capacity, feature_dim,
                ids, cpu_feature_cache_ptr, cache_search_map, gpu_feature_cache_ptr,
                device_id, x, denseC_second_layer)
        nvtx.end_range(rng)

        # softmax 需要 2d matrix
        #x = x.reshape(sparse_matrix_size.f_bcsr_num_rows, self.num_classes)

        return x

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12358'
    # initialize the process group
    if torch.cuda.is_available():
      dist.init_process_group('nccl', rank=rank, world_size=world_size)
    else:
      dist.init_process_group('gloo', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_one_step(model, optimizer, metric, loss_fcn, device, feat_len, iter, device_id, denseC_first_layer, denseC_second_layer):
    # ======================================== 原始采样数据读取 ========================================
    rng = nvtx.start_range(message=("ipc original data"), color="green")
    ids, features, labels, block1_agg_src, block1_agg_dst, block2_agg_src, block2_agg_dst = ipc_service.get_next(feat_len)
    block1_src_num, block1_dst_num, block2_src_num, block2_dst_num = ipc_service.get_block_size()
    nvtx.end_range(rng)

    # ======================================== 混合数据结构读取 ========================================
    rng = nvtx.start_range(message=("ipc mixed data"), color="green")
    sparse_matrix_size = SparseMatrixSize(*tuple(ipc_service.get_sparseMatrix_size()))
    sparse_matrix_data = SparseMatrixData(*tuple(ipc_service.get_sparseMatrix_data()))
    nvtx.end_range(rng)

    # ======================================== feature 信息和数据读取 ========================================
    rng = nvtx.start_range(message=("ipc feature data"), color="green")
    cache_search_map = ipc_service.get_memoryaccess_data()[0] # 这个函数目前只返回了一个 tensor，但是也放到了一个 vector 中，所以类型是 list 而非 tensor
    cpu_float_feature_len, gpu_node_capacity, feature_dim = ipc_service.get_feature_attribute()
    cpu_feature_cache_ptr, gpu_feature_cache_ptr = ipc_service.get_feature_cache_ptr()
    nvtx.end_range(rng)
    
    # ======================================== 计算 ========================================
    rng = nvtx.start_range(message=("train forward"), color="green")
    batch_pred = model(
        sparse_matrix_size,
        sparse_matrix_data,
        cpu_float_feature_len, gpu_node_capacity, feature_dim,
        ids, cpu_feature_cache_ptr, cache_search_map, gpu_feature_cache_ptr,
        device_id,
        denseC_first_layer,
        denseC_second_layer
    )
    batch_pred = batch_pred[:labels.shape[0]]
    nvtx.end_range(rng)

    rng = nvtx.start_range(message=("train compute loss"), color="green")
    long_labels = torch.as_tensor(labels, dtype=torch.long, device=device)
    loss = loss_fcn(batch_pred, long_labels)
    nvtx.end_range(rng)

    optimizer.zero_grad() # 清空梯度(.grad)，避免累积

    rng = nvtx.start_range(message=("train backward"), color="green")
    loss.backward() # 反向传播
    nvtx.end_range(rng)

    rng = nvtx.start_range(message=("train parameters update"), color="green")
    optimizer.step() # 权重更新(基于 optimizer 设置的优化器对参数进行更新)
    nvtx.end_range(rng)

    rng = nvtx.start_range(message=("metric update"), color="green")
    metric.update(batch_pred, long_labels)
    nvtx.end_range(rng)

    """
    # ======================================== torch mm 计算计时 ========================================
    #feature_lookup_start = perf_counter()
    elapsed_feature_lookup = ipc_service.simulate_CacheLookupOP()
    #feature_lookup_end = perf_counter()
    #elapsed_feature_lookup = feature_lookup_end - feature_lookup_start 

    if device_id == 0:
        print(block1_agg_dst.device, block1_agg_dst.shape, block1_agg_dst.is_cuda)
        print(block1_agg_src.device, block1_agg_src.shape, block1_agg_src.is_cuda)
        print("num_rows: {}, num_cols: {}".format(sparse_matrix_size.bcsr_num_rows, sparse_matrix_size.bcsr_num_cols), flush=True)
    indices = torch.stack([block1_agg_dst, block1_agg_src]) # (2, N) 的张量
    values = torch.ones(indices.shape[1]).to(device)  # 构造出来
    A = torch.sparse_coo_tensor(indices, values, size=(sparse_matrix_size.bcsr_num_rows, sparse_matrix_size.bcsr_num_cols))

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    torch.sparse.mm(A, features)
    end.record()
    torch.cuda.synchronize()
    elapsed_spmm = start.elapsed_time(end)

    # 文件记录时间
    save_path = os.path.join("/work/gaohy/Legion_fuse/result/2e4_100_100/cusparse", "")
    file_name = str(device_id) + ".csv"
    file_path = os.path.join(save_path, file_name)
    
    with open(file_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([elapsed_feature_lookup, elapsed_spmm, elapsed_feature_lookup + elapsed_spmm])
    # ======================================== torch SpMM 计算计时 ========================================
    """

    # 一个 batch 的计算完成后，需要将 AX 的结果 free，避免显存占用持续增大
    rng = nvtx.start_range(message=("freeMatC"), color="green")
    #spmm_cnic.freeMatC()
    nvtx.end_range(rng)

    rng = nvtx.start_range(message=("torch sync"), color="green")
    torch.cuda.synchronize()
    nvtx.end_range(rng)

    rng = nvtx.start_range(message=("ipc sync"), color="green")
    ipc_service.synchronize()
    nvtx.end_range(rng)

    return loss

def valid_one_step(model, metric, device, feat_len, device_id):
    ids, features, labels, block1_agg_src, block1_agg_dst, block2_agg_src, block2_agg_dst = ipc_service.get_next(feat_len)
    block1_src_num, block1_dst_num, block2_src_num, block2_dst_num = ipc_service.get_block_size()

    """
    # ======================================== 混合数据结构读取 ========================================
    sparse_matrix_size = SparseMatrixSize(*tuple(ipc_service.get_sparseMatrix_size()))
    sparse_matrix_data = SparseMatrixData(*tuple(ipc_service.get_sparseMatrix_data()))

    # ======================================== feature 信息和数据读取 ========================================
    cache_search_map = ipc_service.get_memoryaccess_data()[0] # 这个函数目前只返回了一个 tensor，但是也放到了一个 vector 中，所以类型是 list 而非 tensor
    cpu_float_feature_len, gpu_node_capacity, feature_dim = ipc_service.get_feature_attribute()
    cpu_feature_cache_ptr, gpu_feature_cache_ptr = ipc_service.get_feature_cache_ptr()

    # ======================================== 计算 ========================================
    batch_pred = model(
        sparse_matrix_size,
        sparse_matrix_data,
        cpu_float_feature_len, gpu_node_capacity, feature_dim,
        ids, cpu_feature_cache_ptr, cache_search_map, gpu_feature_cache_ptr,
        device_id
    )
    batch_pred = batch_pred[:labels.shape[0]]

    long_labels = torch.as_tensor(labels, dtype=torch.long, device=device)

    metric.update(batch_pred, long_labels)

    spmm_cnic.freeMatC()
    """

    ipc_service.synchronize()

    return 0

def test_one_step(model, metric, device, feat_len, device_id):
    ids, features, labels, block1_agg_src, block1_agg_dst, block2_agg_src, block2_agg_dst = ipc_service.get_next(feat_len)
    block1_src_num, block1_dst_num, block2_src_num, block2_dst_num = ipc_service.get_block_size()
    # 注意 Legion 为了适应 dgl 的需求而进行的边反向存储

    """
    # ======================================== 混合数据结构读取 ========================================
    sparse_matrix_size = SparseMatrixSize(*tuple(ipc_service.get_sparseMatrix_size()))
    sparse_matrix_data = SparseMatrixData(*tuple(ipc_service.get_sparseMatrix_data()))

    # ======================================== feature 信息和数据读取 ========================================
    cache_search_map = ipc_service.get_memoryaccess_data()[0] # 这个函数目前只返回了一个 tensor，但是也放到了一个 vector 中，所以类型是 list 而非 tensor
    cpu_float_feature_len, gpu_node_capacity, feature_dim = ipc_service.get_feature_attribute()
    cpu_feature_cache_ptr, gpu_feature_cache_ptr = ipc_service.get_feature_cache_ptr()

    # ======================================== 计算 ========================================
    batch_pred = model(
        sparse_matrix_size,
        sparse_matrix_data,
        cpu_float_feature_len, gpu_node_capacity, feature_dim,
        ids, cpu_feature_cache_ptr, cache_search_map, gpu_feature_cache_ptr,
        device_id
    )
    batch_pred = batch_pred[:labels.shape[0]]

    long_labels = torch.as_tensor(labels, dtype=torch.long, device=device)

    metric.update(batch_pred, long_labels)

    spmm_cnic.freeMatC()
    """

    ipc_service.synchronize()

    return 0

def worker_process(rank, world_size, args):
    print(f"Running GNN Training on CUDA {rank}.")
    device_id = rank
    setup(rank, world_size)
    cuda_device = torch.device("cuda:{}".format(device_id))
    torch.cuda.set_device(cuda_device)

    # 训练与采样一起开始，训练首先进行 model 实例化
    # 调用 GCN class 的 init 方法
    rng = nvtx.start_range(message=("model instantiate"), color="green")
    model = GCN (
        num_features=args.features_num,
        hidden_dim=args.hidden_dim,
        num_classes=args.class_num,
        num_layers=args.hops_num,
        activation=Func.relu,
        dropout=args.drop_rate
    ).to(cuda_device)

    if dist.is_initialized():
        model = DDP(model, device_ids=[device_id])
    nvtx.end_range(rng)

    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # 训练过程所需空间开辟
    rng = nvtx.start_range(message=("initial space malloc and stream"), color="green")
    spmm_cnic.initialize(device_id)

    denseC_size = ipc_service.get_data_scale(device_id)[0]
    denseC_first_layer, denseC_second_layer = spmm_cnic.space_malloc(device_id, denseC_size)
    nvtx.end_range(rng)

    # 等待采样进程 ready
    print(f"device {rank} model instantiate end, waiting for sample ready...", flush=True)
    ipc_service.wait_sample_ready(device_id)
    print(f"device {rank} train begin", flush=True)

    # 初始化过程包含取 IpcHandle，必须等待采样进程处理好之后才能开始
    ipc_service.initialize(world_size)
    train_steps, valid_steps, test_steps = ipc_service.get_steps()
    print("[Info]: train_steps: {}, valid_steps: {}, test_steps: {}".format(train_steps, valid_steps, test_steps))
    
    feat_len = args.features_num
    epoch_num = args.epoch
    for epoch in range(epoch_num):
        # =========================== train ===========================
        print("device-{}, epoch-{} train".format(rank, epoch), flush=True)
        model.train()
        
        # 创建用于计算分类任务准确率的度量指标, 跟踪模型的准确率
        train_metric = torchmetrics.Accuracy('multiclass', num_classes = args.class_num)
        train_metric = train_metric.to(cuda_device)

        #epoch_time = 0
        #epoch_loss = 0.0

        train_steps=50
        start = time.time()
        for iter in range(train_steps):
            rng = nvtx.start_range(message=("train one step(" + str(epoch) + "," + str(iter) + ")"), color="green")
            train_loss = train_one_step(model, optimizer, train_metric, loss_fcn, cuda_device, feat_len, iter, device_id, denseC_first_layer, denseC_second_layer)
            nvtx.end_range(rng)
            #epoch_loss += train_loss.item()
            #if device_id == 0:
            #    print("epoch-{}, iter-{}, loss: {}".format(epoch, iter, train_loss), flush=True)
        #epoch_time += time.time() - start

        #avg_loss = epoch_loss / train_steps
        #acc_train = train_metric.compute().item()
        #if device_id == 0:
        #    print('>>Epoch {} Train Loss :{} Train Accuracy:{}'.format(epoch, avg_loss, acc_train))

        #train_metric.reset()
        
        # =========================== valid ===========================
        print("device-{}, epoch-{} valid".format(rank, epoch), flush=True)

        model.eval()

        # 创建用于计算分类任务准确率的度量指标, 跟踪模型的准确率
        valid_metric = torchmetrics.Accuracy('multiclass', num_classes = args.class_num)
        valid_metric = valid_metric.to(cuda_device)
        
        with torch.no_grad():
            for iter in range(valid_steps):
                rng = nvtx.start_range(message=("valid one step(" + str(epoch) + "," + str(iter) + ")"), color="green")
                #valid_one_step(model, valid_metric, cuda_device, feat_len, device_id)
                nvtx.end_range(rng)
            #acc_val = valid_metric.compute().item()
        
        #if device_id == 0:
        #    print("Epoch:{}, Cost:{} s, Val Acc: {}".format(epoch, epoch_time, acc_val))
        
        #valid_metric.reset()

    # =========================== test ===========================
    print("device-{}, epoch-{} test".format(rank, epoch), flush=True)

    model.eval()

    # 创建用于计算分类任务准确率的度量指标, 跟踪模型的准确率
    test_metric = torchmetrics.Accuracy('multiclass', num_classes = args.class_num)
    test_metric = test_metric.to(cuda_device)

    with torch.no_grad():
        for iter in range(test_steps):
            rng = nvtx.start_range(message=("test one step(" + str(epoch) + "," + str(iter) + ")"), color="green")
            #test_one_step(model, test_metric, cuda_device, feat_len, device_id)
            nvtx.end_range(rng)
        #acc_test = test_metric.compute().item()

    #if device_id == 0:
    #    print("Accuracy on test data: {}".format(acc_test))

    #test_metric.reset()

    # 空间释放
    rng = nvtx.start_range(message=("space destroy"), color="green")
    spmm_cnic.finalize(device_id)
    ipc_service.finalize()
    nvtx.end_range(rng)

    cleanup()

def run_distribute(dist_fn, world_size, args):
    mp.spawn(dist_fn,
             args=(world_size, args),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    cur_path = sys.path[0]
    argparser = argparse.ArgumentParser("Train GNN.")
    argparser.add_argument('--class_num', type=int, default=2)
    argparser.add_argument('--features_num', type=int, default=128)
    argparser.add_argument('--hidden_dim', type=int, default=256)
    argparser.add_argument('--hops_num', type=int, default=2)
    argparser.add_argument('--nbrs_num', type=list, default=[25, 10])
    argparser.add_argument('--drop_rate', type=float, default=0.5)
    argparser.add_argument('--learning_rate', type=float, default=0.003)
    argparser.add_argument('--epoch', type=int, default=2)
    argparser.add_argument('--gpu_number', type=int, default=2)
    args = argparser.parse_args()

    world_size = args.gpu_number

    run_distribute(worker_process, world_size, args)