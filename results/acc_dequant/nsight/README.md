# products F=256：原始 Acc 与 INT4 X 的 Nsight Compute 采集

对应实验报告中 `70.881485 ms → 39.530830 ms` 这一组；需要比较的是两个 kernel。原表后面的 `-44.23%` 和 `1.793×` 是由这两个耗时计算出的变化率和加速比，不是另两个 kernel。

## 当前状态与运行方式

机器安装了 Nsight Compute 2025.3.0。驱动设置 `RmProfilingAdminOnly: 1`，当前账号没有免密 sudo。实际试采返回 `ERR_NVGPUCTRPERM`，详见 `permission_probe.log`；当前尚未生成有效 `.ncu-rep`。性能计数器采集需要在本机终端通过管理员权限执行；准备脚本不修改驱动设置。

已准备专用 profiling 入口：上传同一份 A/X、量化 X、预热指定 kernel 10 次，然后用 `cudaProfilerStart/Stop` 仅包围一次目标 SpMM。不会采集量化、整矩阵反量化、正确性参考或 CUDA Graph 中的大量重复调用，也不会覆盖此前六组实验的 CSV。

先在普通用户终端编译（当前工作区已经编译）：

```bash
cd /home/shixingguo/work/GNN/tcc_spmm
mkdir -p build/tmp
TMPDIR="$PWD/build/tmp" cmake -S . -B build
TMPDIR="$PWD/build/tmp" cmake --build build --target benchmark_acc_dequant -j4
```

采集两个 kernel：

```bash
sudo bash /home/shixingguo/work/GNN/tcc_spmm/bench/profile_acc_dequant.sh
```

密码只在本机终端输入，不要发到聊天里。如果管理员已经开放性能计数器，直接 `bash bench/profile_acc_dequant.sh` 即可。也可以在脚本末尾加 `original` 或 `int4`，只采其中一个。

成功后，本目录生成：

- `products_f256_original.ncu-rep`：原始 Acc，FP32 X。
- `products_f256_int4.ncu-rep`：每个 TC block 读取 INT4 X 并反量化。
- 两份 `.details.txt`、`.metrics.csv` 和 `.capture.log`：文本指标及采集日志。

以上 `.ncu-rep` 必须在采集成功后才会出现，不能把采集失败日志当成报告。用 Nsight Compute GUI 2025.3 或更新的兼容版本打开文件。GUI 可以在另一台电脑运行，只需复制两份报告；查看源码映射时还需要本仓库对应源码。

## 在 GUI 中看什么

1. 打开第一份 `.ncu-rep`，将结果添加为 Baseline，再打开另一份作比较。
2. **GPU Speed Of Light**：比较 SM、DRAM 和缓存相关吞吐利用率。Memory 利用率高不一定就是显存带宽饱和，要分清 DRAM/L2/L1。
3. **Memory Workload Analysis**：看 DRAM 读写字节、吞吐率、L2 命中率及访问效率。重点检查 INT4 是否确实减少 DRAM 读取；不要仅根据存储压缩比推断流量。
4. **Compute Workload Analysis / Instruction Statistics**：看 Tensor Core 和整数指令相关负载，区分 MMA 运算与解包、地址计算开销。
5. **Scheduler / Warp State / Occupancy**：看活跃和 eligible warp、long scoreboard、barrier 等停顿。如果计算和带宽都没有接近上限，可能是访存延迟、调度或同步限制，不能强行二选一。
6. **Hierarchical Roofline (Tensor Core)**：使用 Tensor Core Roofline 辅助定位。此 kernel 的运算是 TF32 MMA + FP32 累加，不是 INT4 MMA；不要拿普通 FP32 CUDA Core 的峰值替代 Tensor Core 峰值。

## 采集口径

- 同一块 GPU 0，同一份 products CSR 和循环扩展到 F=256 的真实特征，与原实验一致。
- kernel replay；两条路径均使用 `--cache-control all` 清理 replay 的缓存，`--clock-control none` 不改 GPU 时钟。
- 启用 `-lineinfo`，便于查看源码/SASS 对应关系；没有修改原始 Acc 或 INT4 kernel 的计算逻辑。
- Nsight 会插桩、重放和清理缓存，其耗时与原来 CUDA Graph 暖缓存计时不完全同口径。原始 `70.881485/39.530830 ms` 保留为性能结果；报告用于分析原因。
- 原始 Acc 已知同步竞争仍保留。profile 不代表修复或证明无竞争；不能只凭一次 replay 推出所有运行都正确。
