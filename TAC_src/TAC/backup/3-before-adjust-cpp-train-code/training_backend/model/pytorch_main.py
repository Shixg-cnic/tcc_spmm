#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GCN性能基准测试工具 - 完整版
用于测量和比较图卷积神经网络在不同配置下的性能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import os
import sys
import psutil
import gc
import json
from datetime import datetime
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from tabulate import tabulate

# 设置常量
MAX_NODES = 1400000  # 可调整的最大节点数量限制

class GraphConvolution(nn.Module):
    """
    内存优化的图卷积层
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """初始化权重"""
        nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        """
        前向传播
        x: 节点特征矩阵 [N, in_features]
        adj: 归一化邻接矩阵 [N, N]，稀疏格式
        """
        # 图卷积操作: X' = AXW
        support = torch.mm(x, self.weight)
        output = torch.sparse.mm(adj, support)
        if self.bias is not None:
            output = output + self.bias
        return output

class GCN(nn.Module):
    """
    图卷积神经网络模型
    """
    def __init__(self, num_features, hidden_dim, num_classes, num_layers=2, dropout=0.5):
        super(GCN, self).__init__()
        # 第一层
        self.conv1 = GraphConvolution(num_features, hidden_dim)
        
        # 中间层
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.hidden_layers.append(GraphConvolution(hidden_dim, hidden_dim))
        
        # 最后一层
        self.conv_final = GraphConvolution(hidden_dim, num_classes)
        
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        
        # 记录模型结构信息
        self.model_info = {
            'num_features': num_features,
            'hidden_dim': hidden_dim,
            'num_classes': num_classes,
            'num_layers': num_layers,
            'dropout': dropout,
            'total_params': sum(p.numel() for p in self.parameters())
        }

    def forward(self, x, adj):
        """
        前向传播
        x: 节点特征
        adj: 归一化邻接矩阵（稀疏格式）
        """
        # 第一层
        x = self.relu(self.conv1(x, adj))
        x = self.dropout(x)
        
        # 中间层
        for conv in self.hidden_layers:
            x = conv(x, adj)
            x = self.relu(x)
            x = self.dropout(x)
        
        # 最后一层
        x = self.conv_final(x, adj)
        
        return F.log_softmax(x, dim=1)

def get_memory_usage():
    """返回当前内存使用量（MB）"""
    if torch.cuda.is_available():
        # GPU内存使用
        return torch.cuda.memory_allocated() / (1024 * 1024)
    else:
        # CPU内存使用
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

def safe_normalize_adj(edge_index, edge_values, num_nodes):
    """
    安全地计算归一化邻接矩阵，使用COO格式避免内存溢出
    返回归一化的边和性能指标
    
    参数:
    edge_index: 边索引 [2, E]
    edge_values: 边权重 [E]
    num_nodes: 节点数量
    
    返回:
    norm_edge_index: 归一化后的边索引
    norm_edge_values: 归一化后的边权重
    metrics: 性能指标字典
    """
    start_time = time.perf_counter()
    mem_before = get_memory_usage()
    
    # 1. 计算度数 - 使用稀疏操作
    # 创建临时COO矩阵用于计算度数
    temp_adj = torch.sparse.FloatTensor(
        edge_index, 
        edge_values,
        torch.Size([num_nodes, num_nodes])
    )
    
    # 每个节点的度数是其所有边的权重和
    degrees = torch.sparse.sum(temp_adj, dim=1).to_dense()
    
    # 2. 计算 D^(-1/2)
    # 对于度数为0的节点，设置为0而不是无穷大
    deg_inv_sqrt = torch.zeros_like(degrees)
    deg_mask = degrees > 0
    deg_inv_sqrt[deg_mask] = degrees[deg_mask].pow(-0.5)
    
    # 3. 计算归一化边权重：每条边(i,j)的新权重 = 旧权重 * (deg_inv_sqrt[i] * deg_inv_sqrt[j])
    src, dst = edge_index[0], edge_index[1]
    normalized_values = edge_values * deg_inv_sqrt[src] * deg_inv_sqrt[dst]
    
    elapsed_time = time.perf_counter() - start_time
    mem_after = get_memory_usage()
    mem_used = mem_after - mem_before
    
    # 4. 计算性能指标
    metrics = {
        'normalization_time': elapsed_time,
        'memory_delta_mb': mem_used,
        'max_degree': degrees.max().item(),
        'avg_degree': degrees.mean().item(),
        'isolated_nodes': (degrees == 0).sum().item()
    }
    
    # 5. 返回归一化后的边、权重和性能指标
    return edge_index, normalized_values, metrics

def load_edges_in_batches(file_path, max_nodes=MAX_NODES):

    start_time = time.perf_counter()
    mem_before = get_memory_usage()
    
    edges = []
    max_node_id = -1
    skipped_edges = 0
    valid_edges = 0
    parsed_lines = 0
    error_lines = 0
    
    print(f"Loading graph data from {file_path}...")
    
    try:
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        # 节点ID从1开始，减1使其从0开始
                        src = int(parts[0].strip()) - 1
                        dst = int(parts[1].strip()) - 1
                        
                        # 检查节点ID是否在允许范围内
                        if 0 <= src < max_nodes and 0 <= dst < max_nodes:
                            edges.append([src, dst])
                            edges.append([dst, src])  # 添加反向边
                            max_node_id = max(max_node_id, src, dst)
                            valid_edges += 1
                        else:
                            skipped_edges += 1
                        
                        parsed_lines += 1
                        
                        # 定期打印进度
                        if line_num % 1000000 == 0:
                            print(f"  Processed {line_num:,} lines, found {valid_edges:,} valid edges")
                    
                except Exception as e:
                    error_lines += 1
                    if error_lines < 10:  # 只显示前几个错误
                        print(f"  Warning: Could not parse line {line_num}: '{line.strip()[:50]}...' - {e}")
                    
    except Exception as e:
        print(f"Error reading file: {e}")
        if not edges:
            raise RuntimeError(f"Could not read any valid edges from {file_path}")
    
    num_nodes = min(max_node_id + 1, max_nodes)
    
    elapsed_time = time.perf_counter() - start_time
    mem_after = get_memory_usage()
    mem_used = mem_after - mem_before
    
    # 收集性能指标
    metrics = {
        'loading_time': elapsed_time,
        'memory_delta_mb': mem_used,
        'total_lines': parsed_lines + error_lines,
        'parsed_lines': parsed_lines,
        'error_lines': error_lines,
        'valid_edges': valid_edges,
        'skipped_edges': skipped_edges,
        'nodes_found': max_node_id + 1,
        'nodes_used': num_nodes,
        'loading_speed': parsed_lines / elapsed_time if elapsed_time > 0 else 0
    }
    
    print(f"Graph loaded: {num_nodes:,} nodes, {valid_edges:,} edges (original direction)")
    print(f"  Loading time: {elapsed_time:.2f} seconds")
    print(f"  Loading speed: {metrics['loading_speed']:.1f} lines/second")
    if skipped_edges > 0:
        print(f"  Skipped {skipped_edges:,} edges due to node limit")
    
    return edges, num_nodes, metrics

def prepare_sparse_graph(edge_list, num_nodes, num_features, num_classes):
    """
    准备图数据，保持稀疏格式
    返回图数据和性能指标
    
    参数:
    edge_list: 边列表 [[src, dst], ...]
    num_nodes: 节点数量
    num_features: 节点特征维度
    num_classes: 类别数量
    
    返回:
    graph_data: 图数据字典
    graph_stats: 性能指标字典
    """
    start_time = time.perf_counter()
    mem_before = get_memory_usage()
    
    print(f"Preparing graph data structures...")
    
    # 1. 准备边索引和值
    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    edge_values = torch.ones(edge_index.size(1))
    
    # 2. 添加自环边
    print(f"  Adding self-loops for {num_nodes} nodes...")
    self_loops_index = torch.arange(num_nodes).repeat(2, 1)
    self_loops_values = torch.ones(num_nodes)
    
    # 3. 合并所有边
    all_edge_index = torch.cat([edge_index, self_loops_index], dim=1)
    all_edge_values = torch.cat([edge_values, self_loops_values])
    
    # 4. 归一化邻接矩阵
    print(f"  Normalizing adjacency matrix...")
    norm_edge_index, norm_edge_values, norm_stats = safe_normalize_adj(
        all_edge_index, all_edge_values, num_nodes
    )
    
    # 5. 创建节点特征（随机）
    print(f"  Generating random node features ({num_nodes} × {num_features})...")
    x = torch.randn((num_nodes, num_features))
    
    # 6. 创建节点标签（随机）
    y = torch.randint(0, num_classes, (num_nodes,))
    
    # 7. 创建训练/验证/测试掩码
    train_ratio = 0.8
    val_ratio = 0.1
    
    indices = torch.randperm(num_nodes)
    train_size = int(train_ratio * num_nodes)
    val_size = int(val_ratio * num_nodes)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size+val_size]] = True
    test_mask[indices[train_size+val_size:]] = True
    
    # 8. 计算性能指标
    elapsed_time = time.perf_counter() - start_time
    mem_after = get_memory_usage()
    mem_used = mem_after - mem_before
    
    # 9. 移动数据到设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Moving data to {device}...")
    
    # 10. 构建图数据
    graph_data = {
        'x': x.to(device),
        'y': y.to(device),
        'edge_index': norm_edge_index.to(device),
        'edge_values': norm_edge_values.to(device),
        'num_nodes': num_nodes,
        'num_edges': len(edge_list) // 2,  # 除以2因为我们添加了反向边
        'train_mask': train_mask.to(device),
        'val_mask': val_mask.to(device),
        'test_mask': test_mask.to(device),
        'num_features': num_features
    }
    
    # 11. 收集性能指标
    graph_stats = {
        'preparation_time': elapsed_time,
        'memory_delta_mb': mem_used,
        'edge_density': len(edge_list) / (num_nodes * num_nodes),
        'avg_degree': len(edge_list) / num_nodes,
        'feature_memory_mb': x.element_size() * x.nelement() / (1024 * 1024),
        'adjacency_memory_mb': (norm_edge_values.element_size() * norm_edge_values.nelement() + 
                               norm_edge_index.element_size() * norm_edge_index.nelement()) / (1024 * 1024),
        'total_edges_with_selfloops': all_edge_index.size(1),
        'split_ratio': f"{train_ratio:.1f}/{val_ratio:.1f}/{1.0-train_ratio-val_ratio:.1f}",
        **norm_stats
    }
    
    print(f"Graph preparation completed in {elapsed_time:.2f} seconds")
    print(f"  Memory used: {mem_used:.1f} MB")
    print(f"  Data split: {train_size:,} train, {val_size:,} validation, {num_nodes-train_size-val_size:,} test")
    
    return graph_data, graph_stats

def train(model, optimizer, data, collect_metrics=True):
    """
    训练一个轮次
    
    参数:
    model: GCN模型
    optimizer: 优化器
    data: 图数据字典
    collect_metrics: 是否收集详细性能指标
    
    返回:
    loss: 损失值
    forward_time: 前向传播时间
    backward_time: 反向传播时间
    metrics: 性能指标字典（如果collect_metrics=True）
    """
    model.train()
    optimizer.zero_grad()
    
    # 性能监控
    metrics = {}
    if collect_metrics:
        mem_before_forward = get_memory_usage()
    
    # 前向传播
    start_forward = time.perf_counter()
    adj = torch.sparse.FloatTensor(
        data['edge_index'], 
        data['edge_values'],
        torch.Size([data['num_nodes'], data['num_nodes']])
    )
    out = model(data['x'], adj)
    forward_time = time.perf_counter() - start_forward
    
    if collect_metrics:
        mem_after_forward = get_memory_usage()
    
    # 计算损失
    loss = F.nll_loss(out[data['train_mask']], data['y'][data['train_mask']])
    
    # 反向传播
    start_backward = time.perf_counter()
    loss.backward()
    backward_time = time.perf_counter() - start_backward
    
    if collect_metrics:
        mem_after_backward = get_memory_usage()
    
    # 更新参数
    start_update = time.perf_counter()
    optimizer.step()
    update_time = time.perf_counter() - start_update
    
    if collect_metrics:
        mem_after_update = get_memory_usage()
        
        # 收集详细性能指标
        metrics = {
            'forward_time': forward_time,
            'backward_time': backward_time,
            'update_time': update_time,
            'total_time': forward_time + backward_time + update_time,
            'forward_memory_mb': mem_after_forward - mem_before_forward,
            'backward_memory_mb': mem_after_backward - mem_after_forward,
            'update_memory_mb': mem_after_update - mem_after_backward,
            'total_memory_mb': mem_after_update - mem_before_forward,
            'loss': loss.item()
        }
    
    return loss.item(), forward_time, backward_time, metrics

@torch.no_grad()
def test(model, data, collect_metrics=True):
    """
    测试模型性能
    
    参数:
    model: GCN模型
    data: 图数据字典
    collect_metrics: 是否收集详细性能指标
    
    返回:
    train_acc: 训练集准确率
    val_acc: 验证集准确率
    test_acc: 测试集准确率
    metrics: 性能指标字典（如果collect_metrics=True）
    """
    model.eval()
    
    # 性能监控
    metrics = {}
    if collect_metrics:
        start_time = time.perf_counter()
        mem_before = get_memory_usage()
    
    # 前向传播
    adj = torch.sparse.FloatTensor(
        data['edge_index'], 
        data['edge_values'],
        torch.Size([data['num_nodes'], data['num_nodes']])
    )
    out = model(data['x'], adj)
    
    # 计算准确率
    pred = out.argmax(dim=1)
    train_correct = pred[data['train_mask']] == data['y'][data['train_mask']]
    val_correct = pred[data['val_mask']] == data['y'][data['val_mask']]
    test_correct = pred[data['test_mask']] == data['y'][data['test_mask']]
    
    train_acc = train_correct.sum().item() / data['train_mask'].sum().item()
    val_acc = val_correct.sum().item() / data['val_mask'].sum().item()
    test_acc = test_correct.sum().item() / data['test_mask'].sum().item()
    
    if collect_metrics:
        elapsed_time = time.perf_counter() - start_time
        mem_after = get_memory_usage()
        
        # 收集性能指标
        metrics = {
            'inference_time': elapsed_time,
            'memory_delta_mb': mem_after - mem_before,
            'inferences_per_second': data['num_nodes'] / elapsed_time,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'test_accuracy': test_acc
        }
    
    return train_acc, val_acc, test_acc, metrics

def get_system_info():
    """获取系统信息"""
    system_info = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'python_version': sys.version,
        'pytorch_version': torch.__version__,
        'hostname': os.uname().nodename if hasattr(os, 'uname') else 'unknown',
        'timestamp': datetime.now().isoformat(),
    }
    
    if torch.cuda.is_available():
        system_info.update({
            'gpu_name': torch.cuda.get_device_name(0),
            'gpu_count': torch.cuda.device_count(),
            'cuda_version': torch.version.cuda,
            'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3)
        })
    else:
        system_info.update({
            'cpu_count': os.cpu_count(),
            'total_memory_gb': psutil.virtual_memory().total / (1024**3)
        })
    
    return system_info

def run_benchmark(config):
    """
    运行性能基准测试
    
    参数:
    config: 包含基准测试配置的字典
    
    返回:
    benchmark_result: 包含所有性能指标的字典
    """
    # 提取配置参数
    dataset_name = config.get('dataset_name', 'DD')
    hidden_dim = config.get('hidden_dim', 64)
    num_layers = config.get('num_layers', 3)
    epochs = config.get('epochs', 200)
    lr = config.get('lr', 0.01)
    num_features = config.get('num_features', 32)
    num_classes = config.get('num_classes', 2)
    max_nodes = config.get('max_nodes', MAX_NODES)
    collect_detailed_metrics = config.get('collect_metrics', True)
    report_interval = config.get('report_interval', 10)
    
    # 基准测试结果
    benchmark_result = {
        'config': config,
        'system_info': get_system_info(),
        'timestamps': {
            'start': datetime.now().isoformat(),
            'data_loaded': None,
            'model_created': None,
            'training_complete': None,
            'end': None
        },
        'data_metrics': {},
        'training_metrics': {
            'epochs': [],
            'best_val_acc': 0,
            'best_test_acc': 0,
            'best_epoch': 0,
            'peak_memory_mb': 0,
            'total_training_time': 0,
            'avg_epoch_time': 0,
            'convergence_epoch': None,
        },
        'model_info': {}
    }
    
    # 开始计时
    overall_start = time.perf_counter()
    
    try:
        # 1. 加载边数据
        file_path = f"./Dataset/{dataset_name}_A.txt"
        print(f"\n{'='*80}\n[Benchmark] Running test: {config.get('name', 'default')}\n{'='*80}")
        edge_list, num_nodes, loading_stats = load_edges_in_batches(file_path, max_nodes)
        benchmark_result['data_metrics']['loading'] = loading_stats
        benchmark_result['timestamps']['data_loaded'] = datetime.now().isoformat()
        
        # 2. 准备图数据
        data, graph_stats = prepare_sparse_graph(edge_list, num_nodes, num_features, num_classes)
        benchmark_result['data_metrics']['graph'] = graph_stats
        
        # 3. 创建模型
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Creating GCN model with {num_layers} layers, {hidden_dim} hidden dim on {device}...")
        
        model = GCN(
            num_features=data['num_features'],
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_layers=num_layers
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        benchmark_result['model_info'] = model.model_info
        benchmark_result['timestamps']['model_created'] = datetime.now().isoformat()
        
        # 4. 训练和评估
        best_val_acc = 0
        best_test_acc = 0
        best_epoch = 0
        early_stop_patience = 300
        early_stop_counter = 0
        convergence_threshold = 0.001  # 准确率变化小于此值认为收敛
        convergence_epoch = None
        total_train_time = 0
        peak_memory = 0
        epoch_metrics = []
        
        print(f"Training for {epochs} epochs...")
        
        for epoch in range(1, epochs + 1):
            epoch_start = time.perf_counter()
            
            try:
                # 训练
                loss, forward_time, backward_time, train_metrics = train(
                    model, optimizer, data, collect_metrics=collect_detailed_metrics
                )
                
                # 测试
                train_acc, val_acc, test_acc, test_metrics = test(
                    model, data, collect_metrics=collect_detailed_metrics
                )
                
                # 记录最佳结果
                if val_acc > best_val_acc:
                    improvement = val_acc - best_val_acc
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                    best_epoch = epoch
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                    improvement = 0
                
                # 检查收敛
                if epoch > 1 and convergence_epoch is None:
                    prev_val_acc = epoch_metrics[-1]['val_acc']
                    if abs(val_acc - prev_val_acc) < convergence_threshold:
                        convergence_epoch = epoch
                
                # 更新性能指标
                epoch_time = time.perf_counter() - epoch_start
                total_train_time += epoch_time
                current_memory = get_memory_usage()
                peak_memory = max(peak_memory, current_memory)
                
                # 收集本轮次指标
                epoch_data = {
                    'epoch': epoch,
                    'loss': loss,
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                    'test_acc': test_acc,
                    'epoch_time': epoch_time,
                    'current_memory_mb': current_memory,
                    'forward_time': forward_time,
                    'backward_time': backward_time,
                    'improvement': improvement
                }
                
                # 添加详细指标（如果收集）
                if collect_detailed_metrics:
                    epoch_data.update(train_metrics)
                    epoch_data.update(test_metrics)
                
                epoch_metrics.append(epoch_data)
                
                # 打印训练信息
                if epoch % report_interval == 0 or epoch == 1:
                    print(f'Epoch: {epoch:03d}, '
                          f'Loss: {loss:.4f}, '
                          f'Train: {train_acc:.4f}, '
                          f'Val: {val_acc:.4f}, '
                          f'Test: {test_acc:.4f}, '
                          f'Time: {epoch_time:.3f}s')
                
                # 早停
                if early_stop_counter >= early_stop_patience:
                    print(f"Early stopping at epoch {epoch}: no improvement for {early_stop_patience} epochs")
                    break
                
            except RuntimeError as e:
                if 'out of memory' in str(e) or 'can\'t allocate memory' in str(e):
                    print(f"ERROR: Out of memory in epoch {epoch}.")
                    benchmark_result['error'] = {
                        'type': 'OOM',
                        'epoch': epoch,
                        'message': str(e)
                    }
                    # 尝试清理内存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    break
                else:
                    benchmark_result['error'] = {
                        'type': 'Runtime',
                        'epoch': epoch,
                        'message': str(e)
                    }
                    raise
        
        # 5. 记录最终结果
        benchmark_result['training_metrics']['epochs'] = epoch_metrics
        benchmark_result['training_metrics']['best_val_acc'] = best_val_acc
        benchmark_result['training_metrics']['best_test_acc'] = best_test_acc
        benchmark_result['training_metrics']['best_epoch'] = best_epoch
        benchmark_result['training_metrics']['peak_memory_mb'] = peak_memory
        benchmark_result['training_metrics']['total_training_time'] = total_train_time
        benchmark_result['training_metrics']['avg_epoch_time'] = total_train_time / len(epoch_metrics) if epoch_metrics else 0
        benchmark_result['training_metrics']['convergence_epoch'] = convergence_epoch
        benchmark_result['timestamps']['training_complete'] = datetime.now().isoformat()
        
# 6. 计算额外的性能指标
        overall_time = time.perf_counter() - overall_start
        benchmark_result['overall_time'] = overall_time
        benchmark_result['timestamps']['end'] = datetime.now().isoformat()
        
        # 计算加速比和效率指标
        benchmark_result['performance_metrics'] = {
            'throughput_nodes_per_second': data['num_nodes'] / benchmark_result['training_metrics']['avg_epoch_time'],
            'throughput_edges_per_second': data['num_edges'] / benchmark_result['training_metrics']['avg_epoch_time'],
            'setup_to_training_ratio': (loading_stats['loading_time'] + graph_stats['preparation_time']) / total_train_time,
            'memory_efficiency': benchmark_result['training_metrics']['best_val_acc'] / peak_memory if peak_memory > 0 else 0,
            'time_to_best_epoch': sum(e['epoch_time'] for e in epoch_metrics[:best_epoch]),
            'param_efficiency': benchmark_result['training_metrics']['best_val_acc'] / model.model_info['total_params'] * 1e6
        }
        
        # 7. 打印总结
        print("\n" + "="*60)
        print(f"BENCHMARK SUMMARY: {config.get('name', 'default')}")
        print("="*60)
        print(f"Dataset: {dataset_name}, Nodes: {data['num_nodes']:,}, Edges: {data['num_edges']:,}")
        print(f"Model: GCN with {num_layers} layers, {hidden_dim} hidden dim, {model.model_info['total_params']:,} parameters")
        print(f"Best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
        print(f"Best test accuracy: {best_test_acc:.4f}")
        if convergence_epoch:
            print(f"Convergence detected at epoch {convergence_epoch}")
        print(f"Peak memory usage: {peak_memory:.2f} MB")
        print(f"Total training time: {total_train_time:.2f} seconds")
        print(f"Average epoch time: {benchmark_result['training_metrics']['avg_epoch_time']:.4f} seconds")
        print(f"Throughput: {data['num_edges']/benchmark_result['training_metrics']['avg_epoch_time']:,.0f} edges/second")
        print(f"Graph processing speed: {data['num_edges']/(loading_stats['loading_time'] + graph_stats['preparation_time']):,.0f} edges/second")
        print("="*60)
        
    except Exception as e:
        # 记录意外错误
        benchmark_result['error'] = {
            'type': 'Unexpected',
            'message': str(e),
            'traceback': str(sys.exc_info())
        }
        print(f"ERROR: {e}")
        raise
        
    return benchmark_result

def run_comparison_benchmarks():
    """
    运行多种配置的基准测试进行比较
    
    返回:
    comparison_results: 包含所有配置测试结果的字典
    """
    results = {}
    
    # 基础配置
    base_config = {
        'dataset_name': 'DD',
        'hidden_dim': 64,
        'num_layers': 3,
        'epochs': 50,  # 减少轮次以加快比较
        'lr': 0.01,
        'num_features': 32,
        'num_classes': 2,
        'max_nodes': 10000,
        'collect_metrics': True,
        'report_interval': 10
    }
    
    # 测试不同的隐藏层维度
    hidden_dims = [32, 64, 128]
    for dim in hidden_dims:
        config = base_config.copy()
        config['hidden_dim'] = dim
        config['name'] = f"hidden_dim_{dim}"
        print(f"\n\n[Benchmark] Running test with hidden_dim={dim}...")
        results[f"hidden_dim_{dim}"] = run_benchmark(config)
    
    # 测试不同的层数
    layer_counts = [2, 3, 4]
    for layers in layer_counts:
        config = base_config.copy()
        config['num_layers'] = layers
        config['name'] = f"layers_{layers}"
        print(f"\n\n[Benchmark] Running test with num_layers={layers}...")
        results[f"layers_{layers}"] = run_benchmark(config)
    
    # 测试不同的节点数限制（如果适用）
    node_limits = [5000, 10000, 20000]
    for limit in node_limits:
        config = base_config.copy()
        config['max_nodes'] = limit
        config['name'] = f"nodes_{limit}"
        print(f"\n\n[Benchmark] Running test with max_nodes={limit}...")
        results[f"nodes_{limit}"] = run_benchmark(config)
    
    # 比较结果
    print("\n\n" + "="*80)
    print("BENCHMARK COMPARISON RESULTS")
    print("="*80)
    
    # 1. 性能指标表格
    headers = ['Configuration', 'Avg Epoch (s)', 'Peak Mem (MB)', 'Best Val Acc', 'Throughput (e/s)', 'Best Epoch']
    rows = []
    
    for name, result in results.items():
        if 'error' in result:
            # 处理有错误的测试
            row = [name, 'ERROR', 'ERROR', 'ERROR', 'ERROR', 'ERROR']
        else:
            avg_epoch = result['training_metrics']['avg_epoch_time']
            peak_mem = result['training_metrics']['peak_memory_mb']
            best_val = result['training_metrics']['best_val_acc']
            best_epoch = result['training_metrics']['best_epoch']
            throughput = result['performance_metrics']['throughput_edges_per_second']
            
            row = [
                name,
                f"{avg_epoch:.4f}",
                f"{peak_mem:.1f}",
                f"{best_val:.4f}",
                f"{throughput:,.0f}",
                f"{best_epoch}"
            ]
        rows.append(row)
    
    print(tabulate(rows, headers=headers, tablefmt='grid'))
    
    # 2. 图数据处理性能
    headers = ['Configuration', 'Loading (s)', 'Prep (s)', 'Edges/Second', 'Memory Usage (MB)']
    rows = []
    
    for name, result in results.items():
        if 'error' not in result:
            loading_time = result['data_metrics']['loading']['loading_time']
            prep_time = result['data_metrics']['graph']['preparation_time']
            edges = result['data_metrics']['loading']['valid_edges']
            mem_used = result['data_metrics']['graph']['memory_delta_mb']
            speed = edges / (loading_time + prep_time) if (loading_time + prep_time) > 0 else 0
            
            row = [
                name,
                f"{loading_time:.2f}",
                f"{prep_time:.2f}",
                f"{speed:,.0f}",
                f"{mem_used:.1f}"
            ]
            rows.append(row)
    
    print("\n" + tabulate(rows, headers=headers, tablefmt='grid'))
    
    # 3. 训练/测试性能演变
    print("\nTraining Performance Evolution (Epoch 10, 25, last):")
    headers = ['Config', 'Epoch 10 Time', 'Epoch 25 Time', 'Last Epoch Time', 
               'Epoch 10 Acc', 'Epoch 25 Acc', 'Final Acc']
    rows = []
    
    for name, result in results.items():
        if 'error' not in result:
            epochs = result['training_metrics']['epochs']
            
            e10_time = epochs[9]['epoch_time'] if len(epochs) > 9 else "N/A"
            e10_acc = epochs[9]['val_acc'] if len(epochs) > 9 else "N/A"
                
            e25_time = epochs[24]['epoch_time'] if len(epochs) > 24 else "N/A"
            e25_acc = epochs[24]['val_acc'] if len(epochs) > 24 else "N/A"
                
            last_time = epochs[-1]['epoch_time'] if epochs else "N/A"
            last_acc = epochs[-1]['val_acc'] if epochs else "N/A"
                
            row = [
                name,
                f"{e10_time}" if isinstance(e10_time, str) else f"{e10_time:.3f}",
                f"{e25_time}" if isinstance(e25_time, str) else f"{e25_time:.3f}",
                f"{last_time}" if isinstance(last_time, str) else f"{last_time:.3f}",
                f"{e10_acc}" if isinstance(e10_acc, str) else f"{e10_acc:.4f}",
                f"{e25_acc}" if isinstance(e25_acc, str) else f"{e25_acc:.4f}",
                f"{last_acc}" if isinstance(last_acc, str) else f"{last_acc:.4f}"
            ]
            rows.append(row)
    
    print(tabulate(rows, headers=headers, tablefmt='grid'))
    
    # 4. 内存分析
    headers = ['Configuration', 'Forward Mem', 'Backward Mem', 'Features Mem', 'Adjacency Mem']
    rows = []
    
    for name, result in results.items():
        if 'error' not in result and result['training_metrics']['epochs']:
            # 取中间轮次的内存数据，应该最稳定
            mid_epoch = len(result['training_metrics']['epochs']) // 2
            if 'forward_memory_mb' in result['training_metrics']['epochs'][mid_epoch]:
                fwd_mem = result['training_metrics']['epochs'][mid_epoch]['forward_memory_mb']
                bwd_mem = result['training_metrics']['epochs'][mid_epoch]['backward_memory_mb']
                
                feat_mem = result['data_metrics']['graph']['feature_memory_mb']
                adj_mem = result['data_metrics']['graph']['adjacency_memory_mb']
                
                row = [
                    name,
                    f"{fwd_mem:.1f}",
                    f"{bwd_mem:.1f}",
                    f"{feat_mem:.1f}",
                    f"{adj_mem:.1f}"
                ]
                rows.append(row)
    
    print("\n" + tabulate(rows, headers=headers, tablefmt='grid'))
    
    # 5. 找出最佳配置
    best_configs = {
        'fastest': min(
            [n for n, r in results.items() if 'error' not in r],
            key=lambda x: results[x]['training_metrics']['avg_epoch_time']
        ),
        'most_accurate': max(
            [n for n, r in results.items() if 'error' not in r],
            key=lambda x: results[x]['training_metrics']['best_val_acc']
        ),
        'lowest_memory': min(
            [n for n, r in results.items() if 'error' not in r],
            key=lambda x: results[x]['training_metrics']['peak_memory_mb']
        ),
        'fastest_convergence': min(
            [n for n, r in results.items() if 'error' not in r and r['training_metrics']['convergence_epoch'] is not None],
            key=lambda x: results[x]['training_metrics']['convergence_epoch']
        ) if any(r['training_metrics']['convergence_epoch'] is not None for r in results.values() if 'error' not in r) else None
    }
    
    print("\n" + "="*40)
    print("BEST CONFIGURATIONS:")
    print(f"Fastest training: {best_configs['fastest']}")
    print(f"Most accurate: {best_configs['most_accurate']}")
    print(f"Lowest memory usage: {best_configs['lowest_memory']}")
    if best_configs['fastest_convergence']:
        print(f"Fastest convergence: {best_configs['fastest_convergence']}")
    print("="*40)
    
    # 6. 创建性能对比图
    try:
        create_performance_plots(results)
    except Exception as e:
        print(f"Warning: Could not create performance plots: {e}")
    
    # 返回汇总结果
    return {
        'timestamp': datetime.now().isoformat(),
        'system_info': get_system_info(),
        'individual_results': results,
        'best_configs': best_configs
    }

def create_performance_plots(results):
    """
    创建性能对比图
    
    参数:
    results: 包含多个测试结果的字典
    """
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    if not valid_results:
        return
    
    plt.figure(figsize=(20, 15))
    
    # 1. 准确率随时间变化图
    plt.subplot(2, 2, 1)
    for name, result in valid_results.items():
        epochs = [e['epoch'] for e in result['training_metrics']['epochs']]
        val_accs = [e['val_acc'] for e in result['training_metrics']['epochs']]
        plt.plot(epochs, val_accs, marker='o', markersize=3, label=name)
    
    plt.title('Validation Accuracy vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 2. 轮次时间对比图
    plt.subplot(2, 2, 2)
    for name, result in valid_results.items():
        epochs = [e['epoch'] for e in result['training_metrics']['epochs']]
        times = [e['epoch_time'] for e in result['training_metrics']['epochs']]
        plt.plot(epochs, times, marker='x', markersize=3, label=name)
    
    plt.title('Epoch Time vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 3. 内存使用对比图
    plt.subplot(2, 2, 3)
    configs = list(valid_results.keys())
    peak_mems = [result['training_metrics']['peak_memory_mb'] for result in valid_results.values()]
    
    plt.bar(configs, peak_mems)
    plt.title('Peak Memory Usage')
    plt.xlabel('Configuration')
    plt.ylabel('Memory (MB)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    # 4. 准确率/时间权衡图
    plt.subplot(2, 2, 4)
    avg_times = [result['training_metrics']['avg_epoch_time'] for result in valid_results.values()]
    best_accs = [result['training_metrics']['best_val_acc'] for result in valid_results.values()]
    
    plt.scatter(avg_times, best_accs, s=100)
    for i, config in enumerate(configs):
        plt.annotate(config, (avg_times[i], best_accs[i]), 
                     textcoords="offset points", xytext=(5,5), ha='left')
    
    plt.title('Accuracy vs Time Trade-off')
    plt.xlabel('Average Epoch Time (seconds)')
    plt.ylabel('Best Validation Accuracy')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('benchmark_results.png', dpi=300)
    print("Performance plots saved to 'benchmark_results.png'")

def plot_training_curves(result, filename=None):
    """
    为单个基准测试结果绘制训练曲线
    
    参数:
    result: 单个基准测试结果
    filename: 保存图片的文件名（可选）
    """
    if 'error' in result or not result['training_metrics']['epochs']:
        print("Cannot plot training curves: no valid data")
        return
    
    plt.figure(figsize=(15, 10))
    
    # 提取数据
    epochs = [e['epoch'] for e in result['training_metrics']['epochs']]
    train_accs = [e['train_acc'] for e in result['training_metrics']['epochs']]
    val_accs = [e['val_acc'] for e in result['training_metrics']['epochs']]
    test_accs = [e['test_acc'] for e in result['training_metrics']['epochs']]
    losses = [e['loss'] for e in result['training_metrics']['epochs']]
    epoch_times = [e['epoch_time'] for e in result['training_metrics']['epochs']]
    
    # 1. 准确率曲线
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_accs, 'b-', label='Train')
    plt.plot(epochs, val_accs, 'r-', label='Validation')
    plt.plot(epochs, test_accs, 'g-', label='Test')
    
    # 标记最佳验证准确率
    best_epoch = result['training_metrics']['best_epoch']
    best_val_acc = result['training_metrics']['best_val_acc']
    plt.axvline(x=best_epoch, color='k', linestyle='--', alpha=0.5)
    plt.scatter([best_epoch], [best_val_acc], color='r', s=100, marker='*')
    plt.annotate(f'Best: {best_val_acc:.4f}', 
                 (best_epoch, best_val_acc),
                 textcoords="offset points", 
                 xytext=(5,10), 
                 ha='left')
    
    plt.title('Accuracy vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 损失曲线
    plt.subplot(2, 2, 2)
    plt.plot(epochs, losses)
    plt.title('Loss vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    # 3. 轮次时间
    plt.subplot(2, 2, 3)
    plt.plot(epochs, epoch_times)
    plt.title('Epoch Time vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.grid(True, alpha=0.3)
    
    # 4. 准确率变化率（收敛分析）
    plt.subplot(2, 2, 4)
    
    # 计算验证准确率的变化率
    val_acc_changes = [0] + [val_accs[i] - val_accs[i-1] for i in range(1, len(val_accs))]
    plt.plot(epochs, val_acc_changes)
    
    # 标记收敛轮次
    if result['training_metrics']['convergence_epoch'] is not None:
        conv_epoch = result['training_metrics']['convergence_epoch']
        plt.axvline(x=conv_epoch, color='r', linestyle='--', alpha=0.7)
        plt.annotate(f'Convergence: {conv_epoch}', 
                     (conv_epoch, 0),
                     textcoords="offset points", 
                     xytext=(5,10), 
                     ha='left')
    
    plt.title('Validation Accuracy Change vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy Change')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300)
        print(f"Training curves saved to '{filename}'")
    else:
        plt.show()

def main():
    """主函数：运行单个基准测试或比较基准测试"""
    parser = argparse.ArgumentParser(description='GCN Performance Benchmark')
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'compare'],
                       help='Benchmark mode: single run or comparison')
    parser.add_argument('--dataset', type=str, default='DD',
                       help='Dataset name (without _A.txt suffix)')
    parser.add_argument('--hidden-dim', type=int, default=64,
                       help='Hidden dimension size')
    parser.add_argument('--num-layers', type=int, default=3,
                       help='Number of GCN layers')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--max-nodes', type=int, default=1000000,
                       help='Maximum number of nodes to consider')
    parser.add_argument('--features', type=int, default=32,
                       help='Number of node features')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                       help='Output JSON file for results')
    parser.add_argument('--plot', action='store_true',
                       help='Generate training curve plots')
    
    args = parser.parse_args()
    
    # 显示系统信息
    sys_info = get_system_info()
    print("="*60)
    print("SYSTEM INFORMATION")
    print("="*60)
    print(f"Device: {sys_info['device']}")
    if 'gpu_name' in sys_info:
        print(f"GPU: {sys_info['gpu_name']} ({sys_info['gpu_memory_gb']:.1f} GB)")
    else:
        print(f"CPU: {sys_info['cpu_count']} cores, {sys_info['total_memory_gb']:.1f} GB RAM")
    print(f"PyTorch: {sys_info['pytorch_version']}")
    print(f"Python: {sys_info['python_version'].split()[0]}")
    print("="*60)
    
    if args.mode == 'single':
        print(f"Running single benchmark with {args.dataset} dataset...")
        config = {
            'dataset_name': args.dataset,
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers,
            'epochs': args.epochs,
            'lr': 0.01,
            'num_features': args.features,
            'num_classes': 2,
            'max_nodes': args.max_nodes,
            'collect_metrics': True,
            'report_interval': 10,
            'name': f"{args.dataset}_h{args.hidden_dim}_l{args.num_layers}"
        }
        
        result = run_benchmark(config)
        
        # 绘制训练曲线
        if args.plot and 'error' not in result:
            plot_training_curves(result, f"{config['name']}_training_curves.png")
        
        # 将结果保存为JSON
        with open(args.output, 'w') as f:
            # 需要将一些不可序列化的对象转换为字符串
            serializable_result = json.loads(
                json.dumps(result, default=lambda o: str(o))
            )
            json.dump(serializable_result, f, indent=2)
        
        print(f"Benchmark results saved to '{args.output}'")
        
    elif args.mode == 'compare':
        print(f"Running comparison benchmarks with {args.dataset} dataset...")
        
        # 修改基础配置以使用命令行参数
        base_config = {
            'dataset_name': args.dataset,
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers,
            'epochs': args.epochs,
            'lr': 0.01,
            'num_features': args.features,
            'num_classes': 2,
            'max_nodes': args.max_nodes,
            'collect_metrics': True,
            'report_interval': 10
        }
        
        # 替换比较函数中的基础配置
        old_run_comparison = run_comparison_benchmarks
        
        def modified_run_comparison():
            results = {}
            
            # 测试不同的隐藏层维度
            hidden_dims = [32, 64, 128]
            for dim in hidden_dims:
                config = base_config.copy()
                config['hidden_dim'] = dim
                config['name'] = f"hidden_dim_{dim}"
                print(f"\n\n[Benchmark] Running test with hidden_dim={dim}...")
                results[f"hidden_dim_{dim}"] = run_benchmark(config)
            
            # 测试不同的层数
            layer_counts = [2, 3, 4]
            for layers in layer_counts:
                config = base_config.copy()
                config['num_layers'] = layers
                config['name'] = f"layers_{layers}"
                print(f"\n\n[Benchmark] Running test with num_layers={layers}...")
                results[f"layers_{layers}"] = run_benchmark(config)
            
            # 测试不同的节点数限制（如果适用）
            node_limits = [args.max_nodes // 2, args.max_nodes, args.max_nodes * 2]
            for limit in node_limits:
                config = base_config.copy()
                config['max_nodes'] = limit
                config['name'] = f"nodes_{limit}"
                print(f"\n\n[Benchmark] Running test with max_nodes={limit}...")
                results[f"nodes_{limit}"] = run_benchmark(config)
            
            # 使用原始函数处理结果比较
            comparison_output = old_run_comparison.__code__.co_consts[3:]  # 跳过函数定义部分
            exec(comparison_output, globals(), locals())
            
            return results
        
        # 运行修改后的比较函数
        results = modified_run_comparison()
        
        # 将结果保存为JSON
        output_filename = f"{args.dataset}_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_filename, 'w') as f:
            serializable_results = json.loads(
                json.dumps(results, default=lambda o: str(o))
            )
            json.dump(serializable_results, f, indent=2)
        
        print(f"Comparison results saved to '{output_filename}'")

if __name__ == '__main__':
    main()