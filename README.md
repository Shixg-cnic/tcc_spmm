# 纯 Tensor Core SpMM

当前只开发 C++/CUDA 算子 `Y=AX`，不进行全局重排。先建立基于 Acc-SpMM 流水线的 FP32 Row、Column kernel，再融合既有 INT8/INT4 反量化；性能模型留到后续。

## 当前实现

Row-BitTCF 将连续 8 个目标行分组，窗口内源列压缩为 8×8 bitmap tile。FP32 权重实际参与 TF32 MMA，支持正负权重与矩形矩阵；CSR 重复坐标先求和，显式零值移除。

保留 Acc 的共享内存双缓冲、`cp.async`、X 寄存器预取、MMA 映射及寄存器累加。窗口独占输出，支持空窗口、尾行和特征维度尾部；F>512 按 512 维分块。输入输出为 FP32，乘法使用 TF32 精度。

```text
tcc_spmm/
├── CMakeLists.txt
├── include/                  # 格式、MMA、流水线声明
├── src/
│   ├── format/row_convert.cpp
│   └── kernel/row_spmm.cu
├── tests/
│   ├── row_spmm_test.cu       # 原生正确性与计时入口
│   ├── acc_reference.cuh
│   └── acc_reference.cu      # 原始 Acc Row 对照
├── third_party/Acc-SpMM/     # 原始源码与 License
└── results/                 # 历史实测记录
```

## 构建与运行

需要 CMake、C++17 编译器和支持 RTX 5090 的 CUDA Toolkit（12.8 或更新版本）。本机先激活 `gnn` 环境。

```bash
git clone https://github.com/Shixg-cnic/tcc_spmm.git
cd tcc_spmm
mkdir -p build
cd build
cmake ..
make -j

# 内置正确性测试，也可用 ctest --output-on-failure
./row_spmm_test

# MatrixMarket 矩阵、F、重复次数、GPU 编号
./row_spmm_test matrix.mtx 128 100 0

# 使用真实 X：无文件头、row-major FP32 [K,F]
./row_spmm_test matrix.mtx 128 100 0 X.bin
```

已克隆仓库时，直接从 `mkdir -p build` 开始。RTX 5090 使用默认 `sm_120`；其他 GPU 可通过 `-DCMAKE_CUDA_ARCHITECTURES=...` 指定架构。

未提供 X 时生成固定种子的随机 FP32 特征。MatrixMarket 支持 coordinate real/integer/pattern，general 或 symmetric；保留权重，不自动加反向边或自环。

生成 `build/libtcc_spmm.a` 和 `build/row_spmm_test`，默认架构为 `sm_120`。CMake 自动查找 CUDA 及原始 Acc 所需的 cuSPARSE/cuBLAS 头文件，不运行 Python。

## C++ 接口与验证

`include/row_format.cuh` 提供 `canonicalCSR(...)`、`convertRow(...)`、`launchRowSpMM(...)`。调用方负责格式数组、X、Y 的显存分配与传输，并传入 CUDA stream。工程没有 Python、PyTorch 或 ctypes 调用层。

原生测试覆盖带权矩形矩阵、bitmap 最高位、重复坐标、空窗口、尾行，以及 F=1/17/32/64/100/128/256/512/513/544/1024，并与 CPU TF32 数学参考比较。计时使用 CUDA Graph 和 Event，取 5 组中位数，转换时间单列。

对照使用未修改的 Acc `METCFBit::convertFromCSR` 与 `tf32_mma_kernel`，要求格式与结果一致。原始基本 kernel 不支持空窗口，且要求 F=64–512、可被 32 整除，其他尺寸只验证新 kernel。该对照没有启用 Acc 的自适应负载均衡。

此前通过 Python 测得的完整 arxiv 和区域结果保留在 [历史报告](results/row_fp32.md)，并非当前原生入口重新计时的结果。原始源码版本 `4fe8d04ae351a97a9e2a37c6cec7afb253b814ce`，MIT License 保留，第三方示例数据不纳入版本控制。来源：[官方实现](https://github.com/Hyaloid/AccSpMM)、[论文](https://arxiv.org/abs/2501.09251)。

下一步实现 FP32 Column-SWTCF：保留 SuperWindow，取消局部度数排序，验证重复输出行缓存开关。
