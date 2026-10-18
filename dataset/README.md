# GNN 训练用真实数据集

本目录包含真实图结构、原始节点特征、分类标签及训练/验证/测试节点索引。特征保留发布数据的维度与数值，仅转换为 float32；没有随机特征、维度截断、循环填充或预先量化。这里与此前 `data/` 中用于 kernel benchmark 的 CSR 和生成特征分开管理。

## 数据规模

| 数据集及版本 | 节点数 | 原图边数及口径 | 实际存储边条目数 | 原始特征维度 | 类别数 |
|---|---:|---|---:|---:|---:|
| ogbn-arxiv / OGB | 169,343 | 1,166,243 条有向边 | 1,166,243 | 128 | 40 |
| ogbn-products / OGB | 2,449,029 | 61,859,140 条无向边（官方计数） | 123,718,280 | 100 | 47 |
| PubMed / Planetoid | 19,717 | 44,324 条无向边 | 88,648 | 500 | 3 |
| DBLP / CitationFull | 17,716 | 52,867 条无向边 | 105,734 | 1,639 | 4 |
| Cora / Planetoid | 2,708 | 5,278 条无向边 | 10,556 | 1,433 | 7 |

`edge_index.npy` 的形状为 `[2, E]`，第一行是源节点 ID，第二行是目标节点 ID。无向图通常以两个方向存储，所以“无向边数”和“存储条目数”不同。表格中的节点数、存储条目数、特征维度和类别数均对本地文件验证。

- **Arxiv** 保留原始有向边，不自动反向化，也不加自环。此前 benchmark 的 2,484,941 个 CSR 非零元来自另一套图预处理，不能当作原始边数。
- **Products** 按 OGB 加载语义为每条原始边添加反向边，保留原始重复/自环记录；没有人为给所有节点加自环。原始边表可在 `raw/edge.csv.gz` 找到。
- **Cora、PubMed、DBLP** 使用无向并集、去重并去除原有自环，没有添加新自环。
- **DBLP** 是论文节点的同构 CitationFull 图，不是作者—论文—会议的异构 DBLP；Cora 是 2,708 节点的 Planetoid 版本，不是 CoraFull。

## 训练划分

| 数据集 | 训练节点 | 验证节点 | 测试节点 | 来源 |
|---|---:|---:|---:|---|
| ogbn-arxiv | 90,941 | 29,799 | 48,603 | 官方 time 划分 |
| ogbn-products | 196,615 | 39,323 | 2,213,091 | 官方 sales_ranking 划分 |
| PubMed | 60 | 500 | 1,000 | Planetoid public 划分 |
| DBLP | 10,629 | 3,542 | 3,545 | 本项目生成，按类别 60%/20%/20%，随机种子 42 |
| Cora | 140 | 500 | 1,000 | Planetoid public 划分 |

DBLP 原始 NPZ 没有提供官方训练划分，本项目的划分**不是论文官方 split**；每类训练/验证数量向下取整，其余分配测试。Cora、PubMed 是半监督节点分类划分，剩余节点仍参与图计算，但不属于三个监督集合。所有集合的索引范围、唯一性及互斥性均验证。

## 目录和数据格式

```text
dataset/
  README.md
  prepare.py                 # 获取/复用原始数据并转换
  load.py                    # NumPy 内存映射加载器
  validate.py                # SHA-256、数据对齐及前向/反向检查
  validation.json
  <数据集名>/
    metadata.json            # 版本、预处理、形状、划分及所有文件 SHA-256
    raw/                     # 原始发布文件，保留原始特征和标签
    processed/
      x.npy                  # float32 [N, F]，原始节点特征
      y.npy                  # int64 [N]，类别标签，编号从 0 开始
      edge_index.npy         # int64 [2, E]，source -> destination
      train_idx.npy          # int64 [N_train]
      valid_idx.npy          # int64 [N_valid]
      test_idx.npy           # int64 [N_test]
      node_year.npy          # 仅 ogbn-arxiv
```

节点 ID、特征行、标签行保持对齐；Planetoid 按其 `test.index` 还原测试节点顺序。图和特征没有进行模型专用的归一化。需要 GCN 自环/归一化、GAT 预处理或量化时，在训练加载后另行处理。DBLP 保留 NPZ 内的原始属性值，没有额外二值化。

## 加载示例

仓库根目录运行，依赖 NumPy；下面的张量示例另外需要 PyTorch：

```python
import torch
from dataset.load import load_dataset

data = load_dataset('ogbn-products', mmap_mode='c')
x = torch.from_numpy(data['x'])
y = torch.from_numpy(data['y'])
edge_index = torch.from_numpy(data['edge_index'])
train_idx = torch.from_numpy(data['split_idx']['train'])
valid_idx = torch.from_numpy(data['split_idx']['valid'])
test_idx = torch.from_numpy(data['split_idx']['test'])
print(x.shape, edge_index.shape, y.shape)
```

`mmap_mode='c'` 使用写时复制映射，张量修改不会改写磁盘数据。Products 较大，是否整图搬到 GPU 或使用邻居采样由训练程序决定。

## 重建与验证

依赖 NumPy、SciPy、pandas；验证脚本另需 PyTorch。当前环境可使用 `/home/shixingguo/miniforge3/envs/gnn/bin/python`。

```bash
python dataset/prepare.py --ogb-root /home/shixingguo/work/GNN/tcc/data
python dataset/validate.py
```

两个 OGB 数据集复用本机已有的官方原始缓存，复制到本目录后独立保存；Cora、PubMed 从 Planetoid 下载，DBLP 从 Graph2Gauss 下载。重建到其他机器前需用官方 OGB 工具准备原始数据，或携带本目录 `raw/` 文件。`prepare.py --names cora pubmed dblp` 可单独准备小图。

`validate.py` 校验原始和处理后文件的 SHA-256、OGB 特征/标签/边对齐及 Products 反向边，然后用真实特征、边、训练标签执行小规模消息聚合及分类损失反向传播。这是数据可用性检查，不是训练精度复现。

大型 `raw/`、`processed/` 数据文件保存在本机，被本目录 `.gitignore` 排除；Git 跟踪脚本、说明、元数据和验证记录，避免提交数 GB 数据。

## 官方来源

- [OGB 节点分类数据集](https://ogb.stanford.edu/docs/nodeprop/)：Arxiv、Products 的图、特征和官方划分。
- [Planetoid 原始数据](https://github.com/kimiyoung/planetoid/tree/master/data)：Cora、PubMed 的图、特征、标签及公开划分。
- [PyG Planetoid 读取方式](https://raw.githubusercontent.com/pyg-team/pytorch_geometric/master/torch_geometric/io/planetoid.py)：测试节点对齐和 public split。
- [Graph2Gauss 原始数据](https://github.com/abojchevski/graph2gauss/tree/master/data) 与 [PyG CitationFull 说明](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/citation_full.html)：DBLP 的具体版本。
