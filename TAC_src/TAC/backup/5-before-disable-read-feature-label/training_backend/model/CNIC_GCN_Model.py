
import torch
import sys
import math
import time 

from tqdm.std import tqdm
import spmm_cnic


class CNICSPMMFunction_TensorCore(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, row_pointers, column_index, 
                blockPartition, edgeToColumn, edgeToRow, 
                hybrid_type, row_nzr, col_nzr):
        # 保存用于反向传播的张量
        ctx.save_for_backward(row_pointers, column_index,
                            blockPartition, edgeToColumn, edgeToRow, 
                            hybrid_type, row_nzr, col_nzr)
        
        # 使用TensorCore版本进行前向计算
        X_out = spmm_cnic.forward_tensorcore(X, row_pointers, column_index,
                                      blockPartition, edgeToColumn, edgeToRow,
                                      hybrid_type, row_nzr, col_nzr)[0]
        return X_out

    @staticmethod
    def backward(ctx, d_output):
        # 恢复保存的张量
        row_pointers, column_index, blockPartition, edgeToColumn, \
        edgeToRow, hybrid_type, row_nzr, col_nzr = ctx.saved_tensors
        
        # 反向传播也使用TensorCore版本
        d_input = spmm_cnic.forward_tensorcore(d_output, row_pointers, column_index,
                                        blockPartition, edgeToColumn, edgeToRow,
                                        hybrid_type, row_nzr, col_nzr)[0]
        
        return d_input, None, None, None, None, None, None, None, None
    
class CNICSPMMFunction_TensorCore_WithWeights(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weights, row_pointers, column_index,
                blockPartition, edgeToColumn, edgeToRow,
                hybrid_type, row_nzr, col_nzr):
        # 保存用于反向传播的张量
        ctx.save_for_backward(X, weights, row_pointers, column_index,
                            blockPartition, edgeToColumn, edgeToRow,
                            hybrid_type, row_nzr, col_nzr)
        
        # 先进行线性变换
        X_prime = torch.mm(X, weights)
        
        # 再使用TensorCore版本进行SpMM计算
        X_out = spmm_cnic.forward_tensorcore(X_prime, row_pointers, column_index,
                                      blockPartition, edgeToColumn, edgeToRow,
                                      hybrid_type, row_nzr, col_nzr)[0]
        return X_out

    @staticmethod
    def backward(ctx, d_output):
        X, weights, row_pointers, column_index, blockPartition, \
        edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr = ctx.saved_tensors
        
        # 先进行SpMM计算
        d_prime = spmm_cnic.forward_tensorcore(d_output, row_pointers, column_index,
                                        blockPartition, edgeToColumn, edgeToRow,
                                        hybrid_type, row_nzr, col_nzr)[0]
        
        # 计算关于输入的梯度
        d_input = torch.mm(d_prime, weights.transpose(0, 1))
        
        # 计算关于权重的梯度
        d_weights = torch.mm(X.transpose(0, 1), d_prime)
        
        return d_input, d_weights, None, None, None, None, None, None, None, None
    


class CNICSPMMFunction_TensorCore_Fused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weights, row_pointers, column_index,
                blockPartition, edgeToColumn, edgeToRow,
                hybrid_type, row_nzr, col_nzr):
        ctx.save_for_backward(X, weights, row_pointers, column_index,
                            blockPartition, edgeToColumn, edgeToRow,
                            hybrid_type, row_nzr, col_nzr)
                            
        # 假设有一个融合版本的TensorCore实现
        X_out = spmm_cnic.forward_tensorcore_fused(X, weights,
                                             row_pointers, column_index,
                                             blockPartition, edgeToColumn, edgeToRow,
                                             hybrid_type, row_nzr, col_nzr)[0]
        return X_out

    @staticmethod
    def backward(ctx, d_output):
        X, weights, row_pointers, column_index, blockPartition, \
        edgeToColumn, edgeToRow, hybrid_type, row_nzr, col_nzr = ctx.saved_tensors
        
        # 使用融合版本进行反向传播
        d_input, d_weights = spmm_cnic.backward_tensorcore_fused(
            d_output, X, weights, row_pointers, column_index,
            blockPartition, edgeToColumn, edgeToRow,
            hybrid_type, row_nzr, col_nzr)
        
        return d_input, d_weights, None, None, None, None, None, None, None, None
    
class GCNConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, implementation='base'):
        super(GCNConv, self).__init__()
        self.weights = torch.nn.Parameter(torch.randn(input_dim, output_dim))
        self.implementation = implementation

    def forward(self, X, row_pointers, column_index, 
                blockPartition, edgeToColumn, edgeToRow,
                hybrid_type, row_nzr, col_nzr):
        if self.implementation == 'tensorcore':
            return CNICSPMMFunction_TensorCore_WithWeights.apply(
                X, self.weights, row_pointers, column_index,
                blockPartition, edgeToColumn, edgeToRow,
                hybrid_type, row_nzr, col_nzr)
        elif self.implementation == 'tensorcore_fused':
            return CNICSPMMFunction_TensorCore_Fused.apply(
                X, self.weights, row_pointers, column_index,
                blockPartition, edgeToColumn, edgeToRow,
                hybrid_type, row_nzr, col_nzr)
        else:
            return CNICSPMMFunction_TensorCore.apply(
                X, row_pointers, column_index,
                blockPartition, edgeToColumn, edgeToRow,
                hybrid_type, row_nzr, col_nzr)