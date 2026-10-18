
目前这个机器跑 nsys 不太行，只能跑 ncu

# Legion is a GPU-initiated system with multi-GPU cache for large-scale GNN training.
```
$ git clone https://github.com/RC4ML/Legion.git
```

## 数据集问题

当时进行了一些例如去除孤点和添加反向边的操作

## 亲和算法

启用亲和算法后，测试集存在数量为 0 的情况，可能需要考虑一下这三个集合的数据分配方式

[Info]: device-0, valid_batch_size: 511
[Info]: device-1, valid_batch_size: 510
[Info]: device-2, valid_batch_size: 512
[Info]: device-3, valid_batch_size: 511
[Info]: device-4, valid_batch_size: 508
[Info]: device-5, valid_batch_size: 510
[Info]: device-6, valid_batch_size: 510
[Info]: device-7, valid_batch_size: 510
[Info]: device-0, test_batch_size: 0
[Info]: device-1, test_batch_size: 0
[Info]: device-2, test_batch_size: 0
[Info]: device-3, test_batch_size: 0
[Info]: device-4, test_batch_size: 0
[Info]: device-5, test_batch_size: 0
[Info]: device-6, test_batch_size: 0
[Info]: device-7, test_batch_size: 0

## 同步关系

实际上每个算子结束之后不需要立刻同步，因为 bcsr 和 bcsc 之间的计算并没有数据依赖关系，可以并行进行

## 一些实现上的坑点

1. 注意 BCSR 和 BCSC 数据结构相关字段的 Ipc Handle 获取时机，不应当在初始化时就获取，应当在使用前（对应采样进程的生成时间）

2. 训练进程读取数据时注意数据类型

3. cudaIpcGetMemHandle 绑定的是 allocation 起始地址（分配时的 base address），并不会识别 C++ 侧的指针偏移, 如果两个指针本质来自同一个 allocation，那么它们产生的句柄会指向同一块内存（影响训练进程获取 data 数据）

4. 注意 config.h 中的 `#ifdef  transpose_` 会影响 `ROW_WINDOW` 的数值。不过目前 `transpose` 这个名称会和内部库的符号发生冲突，造成编译错误，后期需要更改一下名称，目前先在 .h 中手动 define

5. 实现过程中容易遇到的一个想不清的点是采样和训练时生成的 gpu_feature_cache，在采样进程中设置的是一个 `std::vector<float**>`，但是在训练进程中设置的是一个 `float**`。这个原因是，采样进程在进行 cache fillup 时还没有开设多个线程来对应到每一个 gpu 上，所以它的执行层面是全局，需要处理所有 gpu 的 cache，这也就是最外层的 vector 的用途；而采样进程在进行 ipc 时就已经分配到具体的线程上，即具体的 gpu 上了，所以它的执行层面是一个具体 的 gpu 而非全局

## todo

### gcn py

第一层和其他层均调用 `forward_tensorcore()`，但是会传递一个不同的映射表。

第一层映射表是图上原始节点编号与重编号之间的映射，涉及到 feature 的实际访存；从第二层开始，feature 对应的 B 矩阵就变为了上层的输出 C 矩阵，包含此层在内之后的所有层，都只需要按照重编号后的编号进行计算即可。

不过为了实现上的统一，我们可以认为所有层次都会使用到映射表，这样只需要我们在为不同层次建映射表时采用不同的规则即可。

这个映射表就是 sample_ids，在第一层存储的是实际的采样节点编号，在之后的层次存储的是 [0, sample_ids.size()) 顺次编号

### 数据准备

不同层有一个 MatA 矩阵复用的情况，如果每一层的参数都是采样结果，那么每层的计算都需要进行一次格式转换，但是不同层使用的 MatA 的数据是相同的，因此需要将格式转换的过程抽离出来，但是这就涉及到数据结构的信息上移到 python 层面，需要通过 pybind 暴露出来

### 格式转换

```cpp
op_factory_[op_num_ - 2] = NewSSDIOCompleteOP(op_num_ - 2);
op_factory_[op_num_ - 1] = NewFormatTransformOP(op_num_ - 1);

op_params_[i]->stream = (streams_[(i != op_num_-1) ? i%INTRABATCH_CON : 2]);

if(i % INTRABATCH_CON == 1){
    cudaStreamWaitEvent(streams_[1], events_[i / INTRABATCH_CON * INTRABATCH_CON], 0);
} else if (i % INTRABATCH_CON == 2) { 
    cudaStreamWaitEvent(streams_[2], events_[6], 0); 
}
```

### IPC handle 创建

基本流程：
1. 通过 `OpParams* params` 可以获取到 `IPCEnv`
2. 通过 `IPCEnv` 可以访问到 `shmStruct *shm_` 这个 ipc 空间
3. 只需要传递原始指针即可获取 ipc handle，`      cudaIpcGetMemHandle((cudaIpcMemHandle_t *)&shm_->memHandle[device_id][i][0], new_ids);`

修改：
1. `CUDAIPCEnv` 中新增 api 负责调用 `cudaIpcGetMemHandle`
2. `FormatTransform` 新增 `IPCEnv` 参数
3. `FormatTransform` 实现时，在 BCSR 和 BCSC 生成后通过 env 调用 IPC handle 初始化函数

## Structure

1. Python 绑定层（Pybind11） → forward_tensorcore

负责与 Python 交互，接收 torch::Tensor 并转换为原始指针数据

2. Launcher → launch_spmm_kernel_mixedFormat

负责数据处理，关键逻辑在于数据格式的转换（coo -> bcsr & bcsc)

3. Dispatcher -> tf32_spmm, tf32TransposeCompute

负载均衡调度，调用不同的计算策略

4. kernel -> tf32_computeX128TransposePipe2

具体的 kernel 函数实现

## 1. Hardware 
### Hardware Used in Our Paper
All platforms are bare-metal machines.
Table 1
| Platform | CPU-Info | #sockets | #NUMA nodes | CPU Memory | PCIe | GPUs | NVLinks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DGX-V100 | 96*Intel(R) Xeon(R) Platinum 8163 CPU @2.5GHZ | 2 | 1 | 384GB | PCIe 3.0x16, 4*PCIe switches, each connecting 2 GPUs | 8x16GB-V100 | NVLink Bridges, Kc = 2, Kg = 4 |
| Siton | 104*Intel(R) Xeon(R) Gold 5320 CPU @2.2GHZ | 2 | 2 | 1TB | PCIe 4.0x16, 2*PCIe switches, each connecting 4 GPUs | 8x40GB-A100 | NVLink Bridges, Kc = 4, Kg = 2 |
| DGX-A100 | 128*Intel(R) Xeon(R) Platinum 8369B CPU @2.9GHZ | 2 | 1 | 1TB | PCIe 4.0x16, 4*PCIE switches, each connecting 2 GPUs | 8x80GB-A100 | NVSwitch, Kc = 1, Kg = 8 |

Kc means the number of groups in which GPUs connect each other. And Kg means the number of GPUs in each group.


## 2. Software 
Legion's software is light-weighted and portable. Here we list some tested environment.

1. Nvidia Driver Version: 515.43.04

2. CUDA 11.7

3. GCC/G++ 11.4.0

4. OS: Ubuntu(other linux systems are ok)

5. Intel PCM(according to OS version)
```
$ wget https://download.opensuse.org/repositories/home:/opcm/xUbuntu_18.04/amd64/pcm_0-0+651.1_amd64.deb
```
6. pytorch-cu117, torchmetrics
```
$ pip3 install torch-cu1xx
```
7. dgl 1.1.0
```
$ pip3 install  dgl -f https://data.dgl.ai/wheels/cu1xx/repo.html
```
8. MPI-3.1


## 3. Prepare Datasets and Graph Partitioning
Datasets are from OGB (https://ogb.stanford.edu/), Standford-snap (https://snap.stanford.edu/), and Webgraph (https://webgraph.di.unimi.it/).
Here is an example of preparing datasets for Legion.

### Uk-Union Datasets
Refer to README in dataset directory for more instructions
```
$ bash prepare_datasets.sh
```

### Partition Uk-Union
gpu_num represents all gpu numbers you want to use, Legion will partition the graph according to underlying NVlink topology
Note that this step would consume a large volume of CPU memory.
```
$ python graph_partitioning.py --dataset_name 'ukunion' --gpu_num 2
```

## 4. Build Legion from Source

```
$ bash build.sh
```

## 4. Run Legion
There are three steps to train a GNN model in Legion. In these steps, you need to change to **root** user for PCM. (2024.3.11, to solve PCM bugs for general platforms, I disable PCM for now)
### Step 1. Open msr by root for PCM
```
$ modprobe msr
```
### Step 2. Start Legion Server

```
$ python legion_server.py --dataset_path 'dataset' --dataset_name ukunion --train_batch_size 8000 --fanout [25,10] --gpu_number 2 --epoch 2 --cache_memory 38000000 
```

### Step 3. Run Legion Training
After Legion outputs "System is ready for serving", then start training by: 
```
$ python training_backend/legion_graphsage.py --class_num 2  --features_num 128 --hidden_dim 256 --hops_num 2 --gpu_number 2 --epoch 2
```
I will continusly work on this to improve the running process for easier use.

## Cite this work
If you use it in your paper, please cite our work

```
@inproceedings {sun2023legion,
author = {Jie Sun and Li Su and Zuocheng Shi and Wenting Shen and Zeke Wang and Lei Wang and Jie Zhang and Yong Li and Wenyuan Yu and Jingren Zhou and Fei Wu},
title = {Legion: Automatically Pushing the Envelope of Multi-GPU System for Billion-Scale GNN Training},
booktitle = {2023 USENIX Annual Technical Conference (USENIX ATC 23)},
year = {2023},
pages = {165--179}
}
```

## Future Features of Legion
We will open-source SSD support for Legion in the future.


## nvlink affinity

Topology-guided Cache Placement for Memory Optimization

根据图的结构分布（如邻接性、社团性）对数据进行分区与 cache mapping

实现了空间上的结构亲和性（cache/data affinity），提升 cache 命中

节点或边数据的布局更贴合访问模式 → 减少远程访存、降低通信负载

memory access 的路径更“线性化”或“紧凑” → 缓存命中更高、prefetch 更高效

This work proposes a system-level approach for billion-scale GNN training by introducing (1) a topology-guided cache placement mechanism that enhances data locality across GPUs, and (2) a hybrid data-structure-based operator optimization that restructures memory access paths and reduces redundant data movements.

## Hybrid Data-Structure-Based Operator Optimization

### 混合算子

现有的基于采样的的 GNN 系统都是在采样结束后先把 feature 数据从 host memory 读取至 gpu global memory，然后在一个 batch 的采样以及 feature data fetch 结束之后统一交付给训练过程使用。当采样子图较大时，这种实现方案对于显存空间的压力较大，可能产生 out of memory 的问题，也就是说现有系统无法应对大 batch 的采样-训练任务。

为了解决这个问题，我们设计了一种混合数据结构，共同利用现有的 gpu 缓存系统以及 host memory 存储，对存储区域不同的数据区分处理, 充分利用 host memory 相较于 global memory 空间较大的特点，解决大采样子图为 feature 数据读取带来的压力，接对数据结构的介绍，

因为空间不足而设计这种存储结构，可以考虑在现有系统上增设 ssd 的设计初衷是怎么讲的

如何生成的（伪代码）

### SpMM 

基于 tensor core 实现，所有数据都需要加载至 tensor core。通过实验观察到，现有方案下的 host memory -> global memory 相较于 host memory -> gpu register 的时间要慢 10%~15%（是否需要有实测数据）。改变数据读取路径，减少数据中转次数，进而减少读取时间。显著减少子图构建/临时中间结构的生成与搬运 → 访存链路缩短，延迟下降

冷数据部分存在重复读取的问题，为了降低冷数据部分的重复读取开销，我们更改了冷数据的计算模式，算子在执行时能“选择性”或“按需”读取，避免重复拉取稀疏数据片段

## Hybrid Data-Structure-Based Operator Optimization

1. feature fetch 过程从 sample 过程移到 train 过程
2. 冷热数据划分
3. 计算模式改变

### 混合数据结构介绍

为了增强 feature 数据的访问效率和计算效率，我们基于 Acc 提出了一种自适应缓存情况和适应 tensor core 计算的混合稀疏矩阵存储格式 HybridMatrix。HybridMatrix 在整体上分为热数据和冷数据两个部分, 热数据是指 feature 数据已经被缓存至 gpu 的节点，冷数据是指 feture 数据仍位于 cpu 的节点。冷热数据均采用分块的思想，分别利用 4 个数组来进行存储，热数据按照行主序进行组织，划分为多个 row_window，冷数据按照列主序进行组织，划分为多个 col_window，每个 window 都继续划分为多个 tensor core block。row/colWindowOffset 表示每个 window 中的 tc block 的全局初始偏移， tcOffset 表示每个 tc block 第一个非零元的全局初始偏移量，sparseA2B/C 表示每个非零元所在的真实全局列/行编号，tcLocalBit 采用 bitmap 表示每个 tc block 非零元分布情况。

![alt text](image-1.png)

构建过程首先会根据 feature 缓存情况对非零元进行重组, 

会根据 cache 情况自适应生成

To improve memory access patterns and support heterogeneous node degrees, we introduce a hybrid sparse matrix representation, combining a row-major CSR format for low-degree nodes and a column-major CSC format for high-degree nodes. This layout is encapsulated in a unified data structure HybridMatrix, enabling efficient partition-aware SpMM operations.


### 基于混合数据结构的 tensor core SpMM 实现

对混合数据结构同基于 tensor core 的 SpMM 算子进行融合，将热数据的处理同冷数据的处理利用 cuda stream 进行重叠

在训练过程中使用优化后的算子完成计算

## pipeline

需要接着算子来说