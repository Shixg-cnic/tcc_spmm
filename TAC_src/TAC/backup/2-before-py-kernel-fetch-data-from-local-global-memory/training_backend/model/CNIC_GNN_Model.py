import torch
import sys
import math
import time
from tqdm.std import tqdm
import spmm_cnic

class MessagePassingLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, node_features, edge_weights, row_pointers, column_index, 
                blockPartition, edgeToColumn, edgeToRow, 
                hybrid_type, row_nzr, col_nzr):
        # 保存用于反向传播的张量
        ctx.save_for_backward(node_features, edge_weights, row_pointers, column_index,
                             blockPartition, edgeToColumn, edgeToRow, 
                             hybrid_type, row_nzr, col_nzr)
        
        # 高效消息传递实现 - 使用权重对边信息进行加权
        weighted_messages = spmm_cnic.forward_tensorcore_with_edge_weights(
            node_features, edge_weights, row_pointers, column_index,
            blockPartition, edgeToColumn, edgeToRow,
            hybrid_type, row_nzr, col_nzr)[0]
        
        return weighted_messages

    @staticmethod
    def backward(ctx, d_output):
        node_features, edge_weights, row_pointers, column_index, blockPartition, \
        edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr = ctx.saved_tensors
        
        # 反向传播 - 计算节点特征和边权重的梯度
        d_node_features, d_edge_weights = spmm_cnic.backward_tensorcore_with_edge_weights(
            d_output, node_features, edge_weights, row_pointers, column_index,
            blockPartition, edgeToColumn, edgeToRow,
            hybrid_type, row_nzr, col_nzr)
        
        return d_node_features, d_edge_weights, None, None, None, None, None, None, None, None

class GNNLayer(torch.nn.Module):
    """
    通用GNN层，支持多种消息传递和聚合方式
    """
    def __init__(self, in_features, out_features, aggregation='sum', 
                 activation=torch.nn.ReLU(), use_edge_features=False):
        super(GNNLayer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.aggregation = aggregation
        self.activation = activation
        self.use_edge_features = use_edge_features
        
        # 节点特征变换
        self.node_transform = torch.nn.Linear(in_features, out_features)
        
        # 边特征处理（如果启用）
        if use_edge_features:
            self.edge_transform = torch.nn.Linear(in_features, in_features)
        
        # 聚合后的变换
        self.post_transform = torch.nn.Linear(out_features, out_features)
        
        # 跳跃连接（残差连接）
        self.has_residual = (in_features == out_features)
        if in_features != out_features:
            self.residual_mapping = torch.nn.Linear(in_features, out_features)

    def forward(self, X, edge_features=None, row_pointers=None, column_index=None, 
                blockPartition=None, edgeToColumn=None, edgeToRow=None,
                hybrid_type=None, row_nzr=None, col_nzr=None):
        
        # 1. 节点特征变换
        transformed_nodes = self.node_transform(X)
        
        # 2. 边特征处理（可选）
        if self.use_edge_features and edge_features is not None:
            edge_weights = self.edge_transform(edge_features)
        else:
            # 如果没有边特征，使用单位权重
            num_edges = column_index.size(0)
            edge_weights = torch.ones(num_edges, self.in_features, device=X.device)
        
        # 3. 消息传递和聚合
        messages = MessagePassingLayer.apply(
            X, edge_weights, row_pointers, column_index,
            blockPartition, edgeToColumn, edgeToRow,
            hybrid_type, row_nzr, col_nzr)
        
        # 聚合方式选择
        if self.aggregation == 'sum':
            # Sum已经在SpMM中完成
            aggregated = messages
        elif self.aggregation == 'mean':
            # 计算每个节点的度数用于平均
            degrees = torch.sparse.sum(
                torch.sparse.FloatTensor(
                    torch.stack([torch.arange(X.size(0)), torch.arange(X.size(0))]),
                    torch.ones(X.size(0)),
                    (X.size(0), X.size(0))
                ),
                dim=1
            ).to_dense()
            # 避免除以零
            degrees = torch.clamp(degrees, min=1.0)
            aggregated = messages / degrees.unsqueeze(1)
        elif self.aggregation == 'max':
            # 最大聚合需要特殊处理，这里简化处理
            aggregated = messages
        else:
            aggregated = messages
            
        # 4. 聚合后处理
        output = self.post_transform(aggregated)
        
        # 5. 残差连接（如果输入输出维度匹配）
        if self.has_residual:
            output = output + X
        elif hasattr(self, 'residual_mapping'):
            output = output + self.residual_mapping(X)
        
        # 6. 激活函数
        if self.activation is not None:
            output = self.activation(output)
            
        return output

class GNN(torch.nn.Module):
    """
    通用图神经网络模型，支持多层GNN结构
    """
    def __init__(self, in_features, hidden_features, out_features, num_layers=2,
                 dropout=0.1, aggregation='sum', use_edge_features=False):
        super(GNN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = torch.nn.Dropout(dropout)
        
        # 构建GNN层
        self.layers = torch.nn.ModuleList()
        
        # 输入层
        self.layers.append(
            GNNLayer(in_features, hidden_features, 
                    aggregation=aggregation,
                    use_edge_features=use_edge_features)
        )
        
        # 隐藏层
        for _ in range(num_layers - 2):
            self.layers.append(
                GNNLayer(hidden_features, hidden_features,
                        aggregation=aggregation,
                        use_edge_features=use_edge_features)
            )
        
        # 输出层
        self.layers.append(
            GNNLayer(hidden_features, out_features,
                    aggregation=aggregation,
                    activation=None,  # 输出层通常不使用激活函数
                    use_edge_features=use_edge_features)
        )
        
        # 附加的预测头（用于节点分类等任务）
        self.prediction_head = torch.nn.Linear(out_features, out_features)
        
    def forward(self, X, edge_features=None, row_pointers=None, column_index=None, 
                blockPartition=None, edgeToColumn=None, edgeToRow=None,
                hybrid_type=None, row_nzr=None, col_nzr=None):
        
        # 通过所有GNN层
        for i, layer in enumerate(self.layers):
            X = layer(X, edge_features, row_pointers, column_index,
                     blockPartition, edgeToColumn, edgeToRow,
                     hybrid_type, row_nzr, col_nzr)
            
            # 在非最终层应用dropout
            if i < self.num_layers - 1:
                X = self.dropout(X)
        
        # 预测头（例如用于节点分类）
        output = self.prediction_head(X)
        
        return output

def train_gnn(model, optimizer, node_features, edge_features, labels, train_mask,
             row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow,
             hybrid_type, row_nzr, col_nzr, num_epochs=100, patience=20):
    """
    训练GNN模型
    """
    model.train()
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # 前向传播
        start_time = time.time()
        optimizer.zero_grad()
        
        output = model(node_features, edge_features, row_pointers, column_index,
                      blockPartition, edgeToColumn, edgeToRow, 
                      hybrid_type, row_nzr, col_nzr)
        
        # 计算损失（仅对训练节点）
        loss = torch.nn.functional.cross_entropy(output[train_mask], labels[train_mask])
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 计算训练时间
        epoch_time = time.time() - start_time
        
        # 打印进度
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Time: {epoch_time:.4f}s")
        
        # 早停
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    return model

def evaluate_gnn(model, node_features, edge_features, labels, mask,
                row_pointers, column_index, blockPartition, edgeToColumn, edgeToRow,
                hybrid_type, row_nzr, col_nzr):
    """
    评估GNN模型
    """
    model.eval()
    with torch.no_grad():
        output = model(node_features, edge_features, row_pointers, column_index,
                      blockPartition, edgeToColumn, edgeToRow,
                      hybrid_type, row_nzr, col_nzr)
        
        # 计算准确率
        pred = output[mask].argmax(dim=1)
        correct = (pred == labels[mask]).sum().item()
        total = mask.sum().item()
        accuracy = correct / total
        
        # 计算损失
        loss = torch.nn.functional.cross_entropy(output[mask], labels[mask])
        
    return accuracy, loss.item()