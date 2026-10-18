import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import time
import numpy as np
from CNIC_GCN_Model import GCNConv

import CNIC_config 

from CNIC_dataset import *
import spmm_cnic

import pdb;

class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, num_layers=2):
        super(GCN, self).__init__()
        # 第一层
        self.conv1 = GCNConv(num_features, hidden_dim, fixed=1)
        
        # 中间层
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.hidden_layers.append(GCNConv(hidden_dim, hidden_dim, fixed=0))
        
        # 最后一层
        self.conv_final = GCNConv(hidden_dim, num_classes, fixed=2)
        
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x, row_pointers, column_index, blockPartition, 
                edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr, output):
        # 第一层
        x = self.relu(self.conv1(x, row_pointers, column_index, blockPartition,
                                edgeToColumn, edgeToRow, hybrid_type, 
                                row_nzr, col_nzr, output))
        x = self.dropout(x)
        
        # 中间层
        for conv in self.hidden_layers:
            x = conv(x, row_pointers, column_index, blockPartition,
                    edgeToColumn, edgeToRow, hybrid_type, 
                    row_nzr, col_nzr, output)
            x = self.relu(x)
            x = self.dropout(x)
        
        # 最后一层
        x = self.conv_final(x, row_pointers, column_index, blockPartition,
                           edgeToColumn, edgeToRow, hybrid_type, 
                           row_nzr, col_nzr, output)
        
        return F.log_softmax(x, dim=1)

def train(model, optimizer, data, row_pointers, column_index, 
          blockPartition, edgeToColumn, edgeToRow, hybrid_type, 
          row_nzr, col_nzr, output):
    model.train()
    optimizer.zero_grad()
    
    # 前向传播
    start = time.perf_counter()
    out = model(data.x, row_pointers, column_index, blockPartition,
                edgeToColumn, edgeToRow, hybrid_type, 
                row_nzr, col_nzr, output)
    forward_time = time.perf_counter() - start
    
    # 计算损失
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    
    # 反向传播
    start = time.perf_counter()
    loss.backward()
    backward_time = time.perf_counter() - start
    
    # 更新参数
    optimizer.step()
    
    return loss.item(), forward_time, backward_time

@torch.no_grad()
def test(model, data, row_pointers, column_index, blockPartition,
         edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr, output):
    model.eval()
    
    # 前向传播
    out = model(data.x, row_pointers, column_index, blockPartition,
                edgeToColumn, edgeToRow, hybrid_type, 
                row_nzr, col_nzr, output)
    
    # 计算在训练集、验证集和测试集上的准确率
    pred = out.argmax(dim=1)
    train_correct = pred[data.train_mask] == data.y[data.train_mask]
    val_correct = pred[data.val_mask] == data.y[data.val_mask]
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    
    train_acc = train_correct.sum().item() / data.train_mask.sum().item()
    val_acc = val_correct.sum().item() / data.val_mask.sum().item()
    test_acc = test_correct.sum().item() / data.test_mask.sum().item()
    
    return train_acc, val_acc, test_acc

def prepare_data(dataset_name, hidden_dim):

    # 1. 加载数据集
    path = f"./Dataset/{dataset_name}.txt"
    try:
        dataset = spmm_cnic_dataset(
            path=path,
            dim=89,  # DD数据集的原始节点特征维度
            num_class=2,  # DD数据集有2个类别
            verbose=True
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {e}")

    # 2. 获取基本信息并验证
    num_nodes = dataset.num_nodes
    num_edges = dataset.num_edges
    column_index = dataset.column_index
    row_pointers = dataset.row_pointers

    if num_nodes <= 0 or num_edges <= 0:
        raise ValueError("Invalid number of nodes or edges")
    if column_index is None or row_pointers is None:
        raise ValueError("Invalid CSR format data")

    # 3. 计算参数
    num_row_windows = (num_nodes + CNIC_config.BLK_H - 1) // CNIC_config.BLK_H

    # 4. 创建GPU张量
    device = torch.device('cuda')
    output = torch.zeros(
        num_nodes * hidden_dim, 
        dtype=torch.float32, 
        device=device
    ).reshape(num_nodes, hidden_dim)

    # 5. 移动CSR数据到GPU
    try:
        column_index = column_index.to(device)
        row_pointers = row_pointers.to(device)
    except Exception as e:
        raise RuntimeError(f"Failed to move data to GPU: {e}")

    # 6. 执行预处理
    pdb.set_trace()
    try:
        processed_tensors = spmm_cnic.preprocess(
            column_index,      # edgeList_tensor
            row_pointers,      # nodePointer_tensor
            num_nodes,        # 节点数量
            num_edges,        # 边数量
            num_row_windows   # 行窗口数量
        )
        blockPartition, edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr = processed_tensors
    except Exception as e:
        raise RuntimeError(f"Preprocess failed: {e}")

    # 保持原有的返回格式
    return {
        'dataset': dataset,
        'row_pointers': row_pointers,
        'column_index': column_index,
        'blockPartition': blockPartition,
        'edgeToColumn': edgeToColumn,
        'edgeToRow': edgeToRow,
        'hybrid_type': hybrid_type,
        'row_nzr': row_nzr,
        'col_nzr': col_nzr,
        'output': output
    }
    
    
def main():
    # 参数设置
    hidden_dim = 64
    num_layers = 3
    epochs = 200
    lr = 0.01
    
    # 加载数据

    data = prepare_data('DD_A', hidden_dim)
    # 假设数据已经按前面的格式准备好
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = GCN(spmm_cnic_dataset.num_features, hidden_dim, 
                spmm_cnic_dataset.num_classes, num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 记录最好的验证集结果
    best_val_acc = 0
    best_test_acc = 0
    
    # 训练循环
    for epoch in range(1, epochs + 1):
        # 训练
        loss, forward_time, backward_time = train(model, 
                                                  optimizer, 
                                                  data['dataset'],
                                                  data['row_pointers'], 
                                                  data['column_index'],
                                                  data['blockPartition'], 
                                                  data['edgeToColumn'],
                                                  data['edgeToRow'], 
                                                  data['hybrid_type'],
                                                  data['row_nzr'], 
                                                  data['col_nzr'], 
                                                  data['output'])
        
        # 测试
        train_acc, val_acc, test_acc = test(model, 
                                            data['dataset'],
                                            data['row_pointers'], 
                                            data['column_index'],
                                            data['blockPartition'], 
                                            data['edgeToColumn'],
                                            data['edgeToRow'], 
                                            data['hybrid_type'],
                                            data['row_nzr'], 
                                            data['col_nzr'], 
                                            data['output'])
            
        
        # 更新最好的结果
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
        
        # 打印训练信息
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {train_acc:.4f}, '
                  f'Val: {val_acc:.4f}, '
                  f'Test: {test_acc:.4f}, '
                  f'Forward Time: {forward_time*1000:.2f}ms, '
                  f'Backward Time: {backward_time*1000:.2f}ms')
    
    print(f'Best validation accuracy: {best_val_acc:.4f}')
    print(f'Best test accuracy: {best_test_acc:.4f}')

if __name__ == '__main__':
    main()