import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from tqdm import tqdm

class CustomSpMMFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, node_features, adj_indices, adj_values, num_nodes, use_edge_weights=False, edge_weights=None):
        """
        前向传播实现
        node_features: 节点特征矩阵 [num_nodes, feature_dim]
        adj_indices: 邻接矩阵的COO格式索引 [2, num_edges]
        adj_values: 邻接矩阵的COO格式值 [num_edges]
        num_nodes: 节点数量
        use_edge_weights: 是否使用边权重
        edge_weights: 可选的边权重矩阵
        """
        ctx.num_nodes = num_nodes
        ctx.use_edge_weights = use_edge_weights
        
        # 创建稀疏邻接矩阵
        adj_sparse = torch.sparse.FloatTensor(
            adj_indices, 
            adj_values if not use_edge_weights else edge_weights,
            (num_nodes, num_nodes)
        )
        
        # 保存用于反向传播的张量
        ctx.save_for_backward(node_features, adj_indices, adj_values, edge_weights)
        
        # 执行稀疏矩阵乘法: A * X
        output = torch.sparse.mm(adj_sparse, node_features)
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播实现
        """
        node_features, adj_indices, adj_values, edge_weights = ctx.saved_tensors
        num_nodes = ctx.num_nodes
        use_edge_weights = ctx.use_edge_weights
        
        # 创建转置的稀疏邻接矩阵
        adj_transposed_indices = torch.stack([adj_indices[1], adj_indices[0]], dim=0)
        adj_transposed_values = adj_values
        
        adj_transposed = torch.sparse.FloatTensor(
            adj_transposed_indices,
            adj_transposed_values if not use_edge_weights else edge_weights,
            (num_nodes, num_nodes)
        )
        
        # 计算节点特征的梯度: A^T * grad_output
        d_node_features = torch.sparse.mm(adj_transposed, grad_output)
        
        # 如果使用边权重，计算边权重的梯度
        d_edge_weights = None
        if use_edge_weights:
            # 收集源节点和目标节点的特征和梯度
            src_nodes, dst_nodes = adj_indices[0], adj_indices[1]
            src_features = node_features[src_nodes]
            dst_grads = grad_output[dst_nodes]
            
            # 计算边权重的梯度
            d_edge_weights = (src_features * dst_grads).sum(dim=1)
        
        return d_node_features, None, None, None, None, d_edge_weights

class OptimizedGraphConvolution(nn.Module):
    """
    优化的图卷积层，使用自定义的SpMM操作
    """
    def __init__(self, in_features, out_features, bias=True, use_edge_features=False):
        super(OptimizedGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_edge_features = use_edge_features
        
        # 节点变换权重
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        
        # 边变换权重（如果需要）
        if use_edge_features:
            self.edge_transform = nn.Linear(in_features, in_features)
        
        # 偏置
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化参数"""
        nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, edge_index, edge_values, edge_features=None):
        """
        前向传播
        x: 节点特征矩阵 [num_nodes, in_features]
        edge_index: 边索引 [2, num_edges]
        edge_values: 边权重 [num_edges]
        edge_features: 可选的边特征 [num_edges, in_features]
        """
        num_nodes = x.size(0)
        
        # 首先进行节点特征变换: X' = XW
        transformed_features = torch.mm(x, self.weight)
        
        # 处理边特征（如果有）
        processed_edge_weights = None
        if self.use_edge_features and edge_features is not None:
            processed_edge_weights = self.edge_transform(edge_features)
        
        # 使用自定义的SpMM操作: Y = A * X'
        output = CustomSpMMFunction.apply(
            transformed_features, 
            edge_index, 
            edge_values, 
            num_nodes,
            self.use_edge_features,
            processed_edge_weights
        )
        
        # 添加偏置（如果有）
        if self.bias is not None:
            output = output + self.bias
            
        return output

class OptimizedGNN(nn.Module):
    """
    使用优化的图卷积的GNN模型
    """
    def __init__(self, in_features, hidden_dim, out_features, num_layers=2, 
                 dropout=0.5, use_edge_features=False):
        super(OptimizedGNN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.use_edge_features = use_edge_features
        
        # 第一层
        self.conv1 = OptimizedGraphConvolution(
            in_features, 
            hidden_dim,
            use_edge_features=use_edge_features
        )
        
        # 中间层
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.hidden_layers.append(
                OptimizedGraphConvolution(
                    hidden_dim, 
                    hidden_dim,
                    use_edge_features=use_edge_features
                )
            )
        
        # 最后一层
        self.conv_final = OptimizedGraphConvolution(
            hidden_dim, 
            out_features,
            use_edge_features=use_edge_features
        )
        
        self.model_info = {
            'num_features': in_features,
            'hidden_dim': hidden_dim,
            'num_classes': out_features,
            'num_layers': num_layers,
            'use_edge_features': use_edge_features,
            'total_params': sum(p.numel() for p in self.parameters())
        }
    
    def forward(self, x, edge_index, edge_values, edge_features=None):
        """
        前向传播
        x: 节点特征 [num_nodes, in_features]
        edge_index: 边索引 [2, num_edges]
        edge_values: 边权重 [num_edges]
        edge_features: 可选的边特征 [num_edges, feature_dim]
        """
        # 第一层
        x = F.relu(self.conv1(x, edge_index, edge_values, edge_features))
        x = self.dropout(x)
        
        # 中间层
        for conv in self.hidden_layers:
            x = F.relu(conv(x, edge_index, edge_values, edge_features))
            x = self.dropout(x)
        
        # 最后一层
        x = self.conv_final(x, edge_index, edge_values, edge_features)
        
        return F.log_softmax(x, dim=1)

def optimize_adjacency_matrix(edge_index, edge_values, num_nodes):
   
    # 1. 添加自环
    self_loops_index = torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)])
    self_loops_values = torch.ones(num_nodes, device=edge_index.device)
    
    edge_index = torch.cat([edge_index, self_loops_index.to(edge_index.device)], dim=1)
    edge_values = torch.cat([edge_values, self_loops_values])
    
    # 2. 计算度数
    adj_sparse = torch.sparse.FloatTensor(
        edge_index, 
        edge_values,
        (num_nodes, num_nodes)
    )
    degrees = torch.sparse.sum(adj_sparse, dim=1).to_dense()
    
    # 3. 计算正则化系数 D^(-1/2)
    deg_inv_sqrt = torch.zeros_like(degrees)
    deg_mask = degrees > 0
    deg_inv_sqrt[deg_mask] = degrees[deg_mask].pow(-0.5)
    
    # 4. 正则化边权重
    src, dst = edge_index[0], edge_index[1]
    norm_values = edge_values * deg_inv_sqrt[src] * deg_inv_sqrt[dst]
    
    return edge_index, norm_values

def train_epoch(model, optimizer, features, edge_index, edge_values, 
               labels, train_mask, edge_features=None):
    """
    训练一个轮次
    """
    model.train()
    optimizer.zero_grad()
    
    # 记录时间
    start_forward = time.time()
    
    # 前向传播
    output = model(features, edge_index, edge_values, edge_features)
    forward_time = time.time() - start_forward
    
    # 计算损失
    loss = F.nll_loss(output[train_mask], labels[train_mask])
    
    # 反向传播
    start_backward = time.time()
    loss.backward()
    backward_time = time.time() - start_backward
    
    # 更新参数
    optimizer.step()
    
    return loss.item(), forward_time, backward_time

@torch.no_grad()
def evaluate(model, features, edge_index, edge_values, labels, 
            train_mask, val_mask, test_mask, edge_features=None):
    """
    评估模型性能
    """
    model.eval()
    
    # 前向传播
    output = model(features, edge_index, edge_values, edge_features)
    
    # 计算准确率
    pred = output.argmax(dim=1)
    train_correct = pred[train_mask] == labels[train_mask]
    val_correct = pred[val_mask] == labels[val_mask]
    test_correct = pred[test_mask] == labels[test_mask]
    
    train_acc = train_correct.sum().item() / train_mask.sum().item()
    val_acc = val_correct.sum().item() / val_mask.sum().item()
    test_acc = test_correct.sum().item() / test_mask.sum().item()
    
    return train_acc, val_acc, test_acc

def benchmark_gnn(dataset_path, num_features=32, hidden_dim=64, num_classes=2,
                 num_layers=3, use_edge_features=False, epochs=200,
                 lr=0.01, max_nodes=10000, device='cuda'):
    """
    GNN基准测试
    """
    # 1. 加载图数据
    print(f"Loading graph from {dataset_path}...")
    edge_list = []
    max_node_id = -1
    
    with open(dataset_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                src = int(parts[0].strip()) - 1
                dst = int(parts[1].strip()) - 1
                
                if 0 <= src < max_nodes and 0 <= dst < max_nodes:
                    edge_list.append([src, dst])
                    max_node_id = max(max_node_id, src, dst)
    
    num_nodes = min(max_node_id + 1, max_nodes)
    num_edges = len(edge_list)
    print(f"Loaded graph with {num_nodes} nodes and {num_edges} edges")
    
    # 2. 准备数据结构
    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    edge_values = torch.ones(edge_index.size(1))
    
    # 3. 优化邻接矩阵
    edge_index, edge_values = optimize_adjacency_matrix(edge_index, edge_values, num_nodes)
    
    # 4. 创建特征和标签
    features = torch.randn(num_nodes, num_features)
    labels = torch.randint(0, num_classes, (num_nodes,))
    
    # 5. 创建训练/验证/测试掩码
    indices = torch.randperm(num_nodes)
    train_size = int(0.8 * num_nodes)
    val_size = int(0.1 * num_nodes)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size+val_size]] = True
    test_mask[indices[train_size+val_size:]] = True
    
    # 6. 生成边特征（如果需要）
    edge_features = None
    if use_edge_features:
        edge_features = torch.randn(edge_index.size(1), num_features)
    
    # 7. 移动数据到设备
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    features = features.to(device)
    edge_index = edge_index.to(device)
    edge_values = edge_values.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)
    if edge_features is not None:
        edge_features = edge_features.to(device)
    
    # 8. 创建模型
    model = OptimizedGNN(
        in_features=num_features,
        hidden_dim=hidden_dim,
        out_features=num_classes,
        num_layers=num_layers,
        use_edge_features=use_edge_features
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 9. 训练循环
    results = {
        'epochs': [],
        'best_val_acc': 0,
        'best_test_acc': 0,
        'best_epoch': 0
    }
    
    for epoch in range(1, epochs + 1):
        # 训练
        loss, forward_time, backward_time = train_epoch(
            model, optimizer, features, edge_index, edge_values,
            labels, train_mask, edge_features
        )
        
        # 评估
        train_acc, val_acc, test_acc = evaluate(
            model, features, edge_index, edge_values, labels,
            train_mask, val_mask, test_mask, edge_features
        )
        
        # 记录结果
        epoch_result = {
            'epoch': epoch,
            'loss': loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'forward_time': forward_time,
            'backward_time': backward_time
        }
        results['epochs'].append(epoch_result)
        
        # 更新最佳结果
        if val_acc > results['best_val_acc']:
            results['best_val_acc'] = val_acc
            results['best_test_acc'] = test_acc
            results['best_epoch'] = epoch
        
        # 打印进度
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {train_acc:.4f}, '
                  f'Val: {val_acc:.4f}, '
                  f'Test: {test_acc:.4f}, '
                  f'Forward: {forward_time*1000:.2f}ms, '
                  f'Backward: {backward_time*1000:.2f}ms')
    
    # 10. 返回结果
    return {
        'model': model,
        'results': results,
        'model_info': model.model_info,
        'data_info': {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'num_features': num_features,
            'num_classes': num_classes
        }
    }

# 用法示例
if __name__ == "__main__":
    # 运行基准测试
    result = benchmark_gnn(
        dataset_path="./Dataset/DD_A.txt",
        num_features=32,
        hidden_dim=64,
        num_classes=2,
        num_layers=3,
        use_edge_features=False,
        epochs=100
    )
    
    # 打印结果
    print("\nBenchmark Complete!")
    print(f"Best validation accuracy: {result['results']['best_val_acc']:.4f} (epoch {result['results']['best_epoch']})")
    print(f"Best test accuracy: {result['results']['best_test_acc']:.4f}")
    print(f"Model parameters: {result['model_info']['total_params']}")