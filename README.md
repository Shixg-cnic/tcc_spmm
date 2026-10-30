# TCC-SpMM

计算 `Y = A × X̂`：X 按源节点量化为 INT8，在 kernel 内反量化并转 TF32，用 Tensor Core 相乘、FP32 累加。支持 F128、F256。

## 计算与负载均衡

每 32 个目标行组成一个 SW，分为四个 8 行 RW。SW32 按 RW 参与掩码分桶，每个 panel 最多 8 个源节点。一个 CTA 使用 4 warp、128 线程，每 warp 负责 32 维特征；F256 使用两个独立的 128 维特征切片。

格式转换按自然 SW 顺序建立任务描述，每个任务覆盖连续的至多 384 个 panel。轻 SW 仍由一个 CTA 完成，直接写 Y；重 SW 被拆给多个 CTA 并行计算。该机制不排序 SW，不重新排列矩阵数据，实际缩短单个 CTA 的主计算循环。

拆分后仍在每个 panel 内复用跨 RW 的 X。每个 CTA 先将 C 累加在寄存器中；重 SW 的任务完成后，将部分和写入 global workspace，只为任务涉及的 RW 分配空间。每个 SW / 特征切片使用一个完成计数器，最后到达的 CTA 在同一个 kernel 内按固定任务顺序归并部分和并写入 Y。计数器通过 acquire/release 原子操作发布完成状态并自动回到零。没有浮点 atomicAdd、等待其他 CTA 的自旋，也不需要单独启动归并 kernel。

384 限制的是每个任务的 panel 数。不同 panel 涉及 1–4 个 RW，实际工作仍有差异；最后一个 CTA 还需执行归并，因此不是等耗时或对所有矩阵都更快的保证。

独占 RW 的部分按 RW0、RW1、RW2、RW3 处理，每批最多 8 个 panel。四个 producer warp 将 A 和 source ID 通过 `cp.async` 搬入两套 shared buffer，计算当前批时发出下一批搬运。跨 RW 的部分按列处理，每个 warp 加载、反量化一次 X，复用于该 panel 涉及的所有 RW。

```text
task = tasks[CTA]                     # 自然 SW 顺序，连续 panel 区间
C[4个RW] = 0                         # 寄存器
计算 task 内独占 RW 的 panel          # A / source ID 异步双缓冲
计算 task 内跨 RW 的 panel            # 复用加载、反量化后的 X
if SW 未拆分:
    C 写回 global Y
else:
    C 写入该任务独有的 global 部分和
    CTA 同步，发布完成计数，CTA 同步
    if 本 CTA 最后完成:
        按固定顺序归并部分和，写回 global Y
```

此前 GPU 1 / RTX 5090 的运行时资源查询记录：F128/F256 均为每线程 64 registers、每 CTA 4624 B shared、0 B local memory、理论最多 8 CTA/SM。每线程仍保存 32 个 FP32 累加值。

## 构建与运行

需要 CMake 3.18+、支持 C++17 的编译器和 CUDA Toolkit。当前 kernel 使用 TF32 Tensor Core 和 `cp.async`，要求 Ampere 或更新架构（SM80+）。默认编译目标为 SM80，适用于 A100 / A800，CUDA 12.2 可以编译。RTX 5090 应显式指定 SM120，并使用 CUDA 12.8 或更新版本。

在另一台服务器上，从 GitHub 拉取源码并重新编译。下面命令适用于 A100 / A800；先让 `nvcc` 位于 `PATH` 中：

```bash
git clone https://github.com/Shixg-cnic/tcc_spmm.git
cd tcc_spmm
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$(command -v nvcc)" \
  -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build -j4
./build/tcc_spmm_run /path/to/ogbn_products_rabbit_rows.mtx 128 0
./build/tcc_spmm_run /path/to/ogbn_products_rabbit_rows.mtx 256 0
```

其他 GPU 应显式指定 `CMAKE_CUDA_ARCHITECTURES`：A100 / A800 为 `80`，RTX 3090 / A6000 为 `86`，RTX 4090 / L40S 为 `89`，H100 / H200 为 `90`，RTX 5090 为 `120`。使用支持目标架构的 CUDA Toolkit；显式指定架构也支持在没有可见 GPU 的节点上编译。

A800 + CUDA 12.2 示例：

```bash
cmake -S . -B build-a800 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.2/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build-a800 -j4
./build-a800/tcc_spmm_run /path/to/matrix.mtx 128 0
```

从旧版更新时，已有构建目录可能仍缓存 `120`，需用 `-DCMAKE_CUDA_ARCHITECTURES=80`（A800）覆盖缓存；更换 CUDA 编译器时使用新的构建目录。架构选项说明见 [CMake 文档](https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html)。

仓库包含完整源码和编译配置，不包含数据集、已编译程序或性能分析文件。将 `.mtx` 单独传到新服务器，并在运行命令中填写实际路径；不需要旧服务器的目录结构或其他 baseline 项目。

最后一个参数是 CUDA 可见设备编号。只有一个默认计算版本，构建得到 `build/tcc_spmm_run` 和 `build/libtcc_spmm.a`。

程序使用固定种子的随机特征，预热 10 次、计时 10 次。计时包括反量化、SpMM 和 kernel 内部分和归并；不包括矩阵读取、格式转换、任务描述生成、上传、输入量化、打包和 workspace 分配。

调用库时使用默认 RW 分桶格式，上传 `Sw32HostFormat` 的数据及任务描述数组，并显式传入 `SpmmWorkspace`：

- `partials`：`Sw32HostFormat::partialCount * 8 * features` 个 float，无需初始化。
- `counters`：`ceil(rows/32) * (features/128)` 个 uint32，首次使用前清零；同 stream 顺序调用时可直接复用。
- `partialCount == 0` 时两者可为空。并发调用需使用独立 workspace；所有缓冲区保留到 stream 执行完成。

## 源码组织与接口

从 `src/main.cu` 的 `run()` 开始阅读，完整顺序为：矩阵读取与 SW32 转换 → 上传矩阵和任务 → 准备 INT8 特征 → 计时启动。矩阵与 workspace 的设备内存归 `src/device_matrix.cuh` 管理；任务生成单独放在 `buildPanelTasks()`；kernel 内按加载计算、写回、归并分段。详细阅读顺序见 [src/README.md](src/README.md)。

CUDA 错误统一由 `checkCuda()` 抛出异常。`launchSpmmInt8()`、`quantizeFeaturesInt8()` 和 `packFeaturesInt8x16()` 现在返回 `void`；调用方应直接调用，旧的 `check(launchSpmmInt8(...))` 改为 `launchSpmmInt8(...)`。启动仍是异步的，没有加入每次调用后的设备同步。特征准备接口只供非零、有效尺寸使用，packed16 要求特征维度为 16 的倍数；SpMM 支持 128/256。调用方负责缓冲区大小和生命周期。

未使用的 `kernel_info.cuh`、资源/占用率查询接口和独立反量化接口已删除；融合 kernel 中的反量化保留。`cudaOccupancyMaxActiveBlocksPerMultiprocessor` 原本只是估算每个 SM 能同时驻留多少个 CTA，不负责负载均衡或启动调度。

2026-10-30 重构验证：同一 CUDA 12.8 编译器、Release 配置，分别生成 SM80 和 SM120。F128/F256 SpMM、量化和打包 kernel 的机器指令及调度编码与重构前一致，寄存器、共享内存和 local memory 资源记录一致；41 个 CPU 格式案例生成的所有格式数组、任务及归并描述逐字节一致。由于测试服务器 CUDA 驱动初始化返回 999，此次未重新执行 GPU 正确性和耗时测试；下列数据是重构前的历史结果。

## 验证与性能

2026-10-29，RTX 5090 GPU 1，`ogbn_products_rabbit_rows.mtx`，2449029 行、126167053 个非零元素。对照为此次负载均衡修改前的异步双缓冲 + SW 组排序版本。同一进程、同一份驻留输入，旧版 / 正式新版交替计时，每轮 8 组 × 每组 20 次，复测两轮。下表为各轮组中位数范围；新版包含完整归并开销。

| 特征维度 | 修改前 | 当前版本 | 耗时下降 |
| --- | ---: | ---: | ---: |
| F128 | 7.332–7.345 ms | 6.643–6.653 ms | 约 9.4% |
| F256 | 14.618–14.647 ms | 13.193–13.211 ms | 约 9.8% |

使用上述普通运行命令，F128 另一次 10 次平均为 6.409 ms；这是单独运行结果，不与交替测试混算。测量期间 GPU 1 存在另一常驻进程，交替复测用于降低环境变化的影响。

该矩阵有 76533 个 SW，其中 1943 个被拆分，总计 78801 个任务；最重 SW 的 2417 个 panel 被拆成 7 个 CTA。部分和空间为 F128 59.2 MiB、F256 118.4 MiB，另有任务描述和完成计数器。空间及归并开销随被拆分的工作量增加；轻 SW 不经过部分和归并。

独立 CPU 参考的 44 个案例全部通过，覆盖 F128/F256、空矩阵、全部 RW mask、尾行/尾 panel、长独占与跨 RW bucket、倾斜负载及 TF32 舍入。使用非默认 nonblocking stream，重复调用并检查输出覆盖、边界 guard 和计数器复位。针对重负载案例分别运行 Compute Sanitizer memcheck、racecheck、synccheck，均为零错误，racecheck 为零警告。

Products 全部输出与旧版按数值容差比较通过。拆分改变 FP32 求和结合顺序，不保证逐位一致；与旧版最大绝对差 F128 为 0.000656128，F256 为 0.000427246。CPU 参考验证使用 TF32 操作数与误差界检查。临时实验版本与检查程序已清理。这些收益只针对上述矩阵，尚不代表其他矩阵或与 Acc 的比较结论。

## 数据与目录

`.mtx` 支持 coordinate real/integer/pattern 以及 general/symmetric；对称输入补齐非对角元素，重复坐标相加。行重排在算子外完成，输出遵循输入矩阵的行顺序。`data/ogbn_products_rabbit_rows.mtx` 为仅重排目标行的 Products，原始矩阵为 `data/ogbn_products_full.mtx`。Rabbit 上游源码在 `../baselines/reordering/RabbitOrder/`。

`src/format/` 负责格式、任务描述与特征，`src/kernel/` 负责 kernel，`src/main.cu` 是运行入口。`data/` 的既有二进制数据继续供其他项目使用。
