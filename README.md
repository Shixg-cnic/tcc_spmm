# 纯 Tensor Core SpMM

当前只开发 C++/CUDA 算子 `Y=AX`，不进行全局重排。先建立基于 Acc-SpMM 流水线的 FP32 Row、Column kernel，再融合既有 INT8/INT4 反量化；性能模型留到后续。

## 当前实现

Row-BitTCF 将连续 8 个目标行分组，窗口内源列压缩为 8×8 bitmap tile。FP32 权重实际参与 TF32 MMA，支持正负权重与矩形矩阵；CSR 重复坐标先求和，显式零值移除。

保留 Acc 的共享内存双缓冲、`cp.async`、X 寄存器预取、MMA 映射及寄存器累加。窗口独占输出，支持空窗口、尾行和特征维度尾部；F>512 按 512 维分块。输入输出为 FP32，乘法使用 TF32 精度。

Column-SWTCF 将连续 8 个输入列分组，按原始目标行编号压缩为 8×8 Tile，默认每 16 个列窗口组成 SuperWindow。所有版本支持实际 FP32 权重，复用窗口内的 X，使用 TF32 MMA，并通过全局原子累加输出。

自适应变体（M3）：column_adaptive.cu 按任务表驱动，grid.x = 任务数；任务由
uildAdaptiveSchedule 的确定性启发式生成（重 CW 拆分为 256-tile chunk，
轻 CW 按复用收益贪心分组），支持 C2/C3 风格缓存开关。调度统计与
可校准性能模型见 ench/model_calibration.py。

| 版本 | CTA 任务 | 稀疏 Tile 流水线 | 重复输出行缓存 |
| --- | --- | --- | --- |
| C0 | Column Window × 128 特征 | 同步加载 | 关闭 |
| C1 | 同 C0 | `cp.async` 双缓冲 | 关闭 |
| C2 | SuperWindow × 128 特征 | 同 C1，增加下一窗口 X 预取 | 关闭 |
| C3 | 同 C2 | 同 C2 | 最多 32 行，SW 结束时写回 |

```text
tcc_spmm/
├── CMakeLists.txt
├── include/                  # 公共 CSR、Row/Column 格式、MMA、流水线声明
│   ├── csr.cuh               # CanonicalCSR 公共定义（Row/Column 共享）
│   ├── row_format.cuh
│   ├── column_format.cuh
│   ├── column_schedule.cuh   # 动态 SW 调度（任务表/自适应格式）
│   ├── mma_tf32.cuh
│   └── pipeline.cuh
├── src/
│   ├── format/               # csr / Row / Column CPU 转换（含 column_schedule.cpp）
│   └── kernel/               # Row / Column FP32 Tensor Core 算子（含 column_adaptive.cu）
├── tests/
│   ├── row_spmm_test.cu       # 原生正确性与计时入口
│   ├── column_spmm_test.cu    # C0–C3 正确性、格式解码、Graph 重放
│   ├── compare_row_column.cu # 同输入性能比较
│   ├── adaptive_sw_test.cu   # 动态调度正确性/覆盖率 + 4 配置 x 11 个 F
│   ├── column_test_utils.cuh
│   ├── matrix_io.cuh
│   ├── acc_reference.cuh
│   └── acc_reference.cu      # 原始 Acc Row 对照
├── bench/
│   ├── benchmark_fp32.cu     # 统一 FP32 基线：Acc 原始 + Row + C0–C3 + Dyn*
│   └── model_calibration.py  # 可校准性能模型（3.6）
├── third_party/Acc-SpMM/     # 原始源码与 License
└── results/                 # 实测报告及原始记录（baseline/adaptive/int4）
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
./column_spmm_test

# MatrixMarket 矩阵、F、重复次数、GPU 编号
./row_spmm_test matrix.mtx 128 100 0

# 使用真实 X：无文件头、row-major FP32 [K,F]
./row_spmm_test matrix.mtx 128 100 0 X.bin

# 同输入比较 Row 和 C0/C1/C2/C3；all 包含 F=32/64/128/256/512
./compare_row_column matrix.mtx all 30 0
./compare_row_column matrix.mtx 128 30 0 X.bin

# 统一 FP32 基线：Acc 原始 tf32_mma_kernel + Row + C0/C1/C2/C3
# + DynCache/DynNoSplit/DynNoCache（方法 6-8）
# all 包含 F=32/64/100/128/256/512；groups>=3 组独立 Graph 测量
./benchmark_fp32 matrix.csr all 3 0 X.bin
./benchmark_fp32 matrix.csr 128 3 0 X.bin

# 内存和共享内存竞争检查
compute-sanitizer --tool memcheck --error-exitcode 9 ./column_spmm_test
compute-sanitizer --tool racecheck --error-exitcode 9 ./column_spmm_test
```

已克隆仓库时，直接从 `mkdir -p build` 开始。RTX 5090 使用默认 `sm_120`；其他 GPU 可通过 `-DCMAKE_CUDA_ARCHITECTURES=...` 指定架构。
Column 默认参数可通过 `cmake .. -DTCC_SW_SIZE=16 -DTCC_CACHE_SIZE=32` 编译配置，转换接口也允许指定运行时 SW 大小。格式与 Kernel 必须使用相同的编译期缓存容量。

未提供 X 时生成固定种子的随机 FP32 特征。MatrixMarket 支持 coordinate real/integer/pattern，general 或 symmetric；保留权重，不自动加反向边或自环。

生成 `build/libtcc_spmm.a` 和四个原生程序（三个测试 + 一个基准），默认架构为 `sm_120`。CMake 自动查找 CUDA 及原始 Acc 所需的 cuSPARSE/cuBLAS 头文件，不运行 Python。

## C++ 接口与验证

`include/csr.cuh` 提供公共 `CanonicalCSR` 与 `canonicalCSR(...)`；`include/row_format.cuh` 提供 `convertRow(...)`、`launchRowSpMM(...)`；`include/column_format.cuh` 提供 `convertColumn(...)`、GPU 数组视图 `DeviceColumnSWTCF`、`launchColumnSpMM(...)` 和资源查询。调用方负责格式数组、X、Y 的显存分配与传输，并传入 CUDA stream。工程没有 Python、PyTorch 或 ctypes 调用层。

```cpp
launchColumnSpMM(deviceColumn, deviceX, deviceY, F, ColumnVersion::C3, stream);
```

默认调用包含同 Stream 上的 `cudaMemsetAsync(Y)`，可直接捕获进 CUDA Graph。仅在已经清零 Y 时使用末尾参数 `clearOutput=false`。GPU 指针由调用者管理，X 与 Y 不得重叠。

`benchmark_fp32` 直接调用原始 Acc `tf32_mma_kernel`（未修改源码），要求与 Column 相同的 CSR/X/F/GPU/stream。Acc 原始 kernel 的已知限制：F 必须在 [64,512] 且 F%32==0；行数填充到 8 的倍数；不支持空 8 行窗口（本仓库两个全图均无空窗口）。这些限制只影响 Acc 对照列，本地 Row/Column 无此限制。计时统一使用 CUDA Graph + Event：Column 的 complete 包含 Y 清零，isolated kernel 与 complete 在同一 Graph 内用外部事件计时，避免测量序列不同导致的时钟状态差异。

`compare_row_column` 同时支持 MatrixMarket 和原生 CSR 二进制文件：8 字节 `TCCCSR1\0`，三个 little-endian uint32（M、K、nnz），然后依次是 uint32 `ptr[M+1]`、uint32 `src[nnz]`、FP32 `values[nnz]`。读取后合并重复坐标。可选 X 文件是 row-major FP32 `[K,sourceF]`；F 较小时截取，F 较大时循环扩展原特征。数据放在被 Git 忽略的 `data/` 下。

原生测试覆盖带权矩形矩阵、bitmap 最高位、重复坐标、空窗口、尾行，以及 F=1/17/32/64/100/128/256/512/513/544/1024，并与 CPU TF32 数学参考比较。计时使用 CUDA Graph 和 Event，取 5 组中位数，转换时间单列。

对照使用未修改的 Acc `METCFBit::convertFromCSR` 与 `tf32_mma_kernel`，要求格式与结果一致。原始基本 kernel 不支持空窗口，且要求 F=64–512、可被 32 整除，其他尺寸只验证新 kernel。该对照没有启用 Acc 的自适应负载均衡。

此前通过 Python 测得的完整 arxiv 和区域结果保留在 [历史报告](results/row_fp32.md)，并非当前原生入口重新计时的结果。原始源码版本 `4fe8d04ae351a97a9e2a37c6cec7afb253b814ce`，MIT License 保留，第三方示例数据不纳入版本控制。来源：[官方实现](https://github.com/Hyaloid/AccSpMM)、[论文](https://arxiv.org/abs/2501.09251)。

第二阶段实现及同输入数据见 [Column FP32 报告](results/column_fp32.md)。该阶段发现并修复了 Row 首次映射缓冲的跨 Warp 竞争，新增一次计算前同步；第三方源码保持原样，历史 Row 数据保留。下一阶段再设计融合反量化。

第三阶段基线（Acc 原始 + Row + C0–C3 统一计时）见 [results/baseline/baseline_fp32.md](results/baseline/baseline_fp32.md)。
M2 列优化分析见 [results/baseline/column_fp32_optimized.md](results/baseline/column_fp32_optimized.md)。
M3 动态 SuperWindow 调度及模型见 [results/baseline/column_fp32_dynamic.md](results/baseline/column_fp32_dynamic.md)
与 [docs/design.md](docs/design.md)。
