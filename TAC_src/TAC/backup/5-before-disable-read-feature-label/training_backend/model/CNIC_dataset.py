import torch
import numpy as np
import time
from scipy.sparse import *

class spmm_cnic_dataset(torch.nn.Module):
    def __init__(self, path, dim, num_class, load_from_txt=True, verbose=False):
        super(spmm_cnic_dataset, self).__init__()
        self.nodes = set()
        self.load_from_txt = load_from_txt
        self.num_nodes = 0
        self.num_features = dim 
        self.num_classes = num_class
        self.edge_index = None
        self.reorder_flag = False
        self.verbose_flag = verbose
        self.avg_degree = -1
        self.avg_edgeSpan = -1

        self.init_edges(path)
        self.init_embedding(dim)
        self.init_labels(num_class)

        # 设置训练、验证、测试掩码
        train = 1
        val = 0.3
        test = 0.1
        self.train_mask = [1] * int(self.num_nodes * train) + [0] * (self.num_nodes - int(self.num_nodes * train))
        self.val_mask = [1] * int(self.num_nodes * val) + [0] * (self.num_nodes - int(self.num_nodes * val))
        self.test_mask = [1] * int(self.num_nodes * test) + [0] * (self.num_nodes - int(self.num_nodes * test))
        self.train_mask = torch.BoolTensor(self.train_mask).cuda()
        self.val_mask = torch.BoolTensor(self.val_mask).cuda()
        self.test_mask = torch.BoolTensor(self.test_mask).cuda()

    def init_edges(self, path):
        if self.load_from_txt:
            fp = open(path, "r")
            src_li = []
            dst_li = []
            start = time.perf_counter()
            
            for line in fp:
                # 移除空白字符并按逗号分割
                src, dst = line.strip().split(",")
                # 转换为整数，并减1转换为0-based索引
                src, dst = int(src) - 1, int(dst) - 1  # 从1-based转换为0-based
                src_li.append(src)
                dst_li.append(dst)
                self.nodes.add(src)
                self.nodes.add(dst)
            
            fp.close()
            
            self.num_edges = len(src_li)
            self.num_nodes = max(self.nodes) + 1
            self.edge_index = np.stack([src_li, dst_li])

            # 计算平均度和边跨度
            self.avg_degree = self.num_edges / self.num_nodes
            self.avg_edgeSpan = np.mean(np.abs(np.subtract(src_li, dst_li)))

            if self.verbose_flag:
                print(f"# Loading (txt) {time.perf_counter() - start:.3f}s")
                print(f'# nodes: {self.num_nodes}')
                print(f"# edges: {self.num_edges}")
                print(f"# avg_degree: {self.avg_degree:.2f}")
                print(f"# avg_edgeSpan: {int(self.avg_edgeSpan)}")

            # 构建图的CSR表示
            start = time.perf_counter()
            val = np.ones(self.num_edges)
            scipy_coo = coo_matrix((val, self.edge_index), 
                                 shape=(self.num_nodes, self.num_nodes))
            scipy_csr = scipy_coo.tocsr()

            if self.verbose_flag:
                print(f"# Build CSR (s): {time.perf_counter() - start:.3f}")

            # 转换为PyTorch张量
            self.column_index = torch.IntTensor(scipy_csr.indices)
            self.row_pointers = torch.IntTensor(scipy_csr.indptr)

            # 计算节点的度
            degrees = (self.row_pointers[1:] - self.row_pointers[:-1]).tolist()
            self.degrees = torch.sqrt(torch.FloatTensor([max(1, d) for d in degrees])).cuda()

        else:
            if not path.endswith('.npz'):
                raise ValueError("graph file must be a .npz file")
            
            start = time.perf_counter()
            graph_obj = np.load(path)
            src_li = graph_obj['src_li']
            dst_li = graph_obj['dst_li']
            
            self.num_nodes = graph_obj['num_nodes']
            self.num_edges = len(src_li)
            self.edge_index = np.stack([src_li, dst_li])
            
            if self.verbose_flag:
                print(f"# Loading (npz)(s): {time.perf_counter() - start:.3f}")

    def init_embedding(self, dim):
        """生成节点的嵌入向量"""
        self.x = torch.randn(self.num_nodes, dim).cuda()
    
    def init_labels(self, num_class):
        """生成节点的标签"""
        self.y = torch.ones(self.num_nodes).long().cuda()