# Tensor Core SpMM：Q1 / 固定 Q4

当前按 2026-10-18 的[新方案](docs/q4-request.txt)推进：显式源列映射 → 32 列轻量分组 → Q4 输出生命周期归约 → 转换外提、float4 和双缓冲 → Acc 完整计时。X、Y 继续使用原始节点编号，CSR 坐标和浮点权重不变。

第二轮已按[审查意见](docs/q4-round2-request.txt)补齐标量双缓冲、H16/H24/H32 分派、A shared XOR 布局及局部 task 扰动，并修正 Sync 的 inline PTX 输入约束。使用[第二轮报告](results/q4_round2/report.md)查看交错计时和五个原始样本；[实现说明](docs/q4-round2.md)记录接口、SASS 与实验边界。

此前的 CW 执行顺序实验、INT4、自适应 SW 和旧 C0–C3 路线停止作为开发主线。代码与结果保留作历史参考，旧说明在 [legacy-readme.md](docs/legacy-readme.md)。

## 当前入口

- `include/column_q4.cuh`：独立的新格式、设备视图和启动接口。
- `src/format/column_q4.cpp`：固定种子 64 位 MinHash；每 32 源列增量 overlap gain 贪心构造 8 列 CW；每组邻接预算 262144，超限回退；tileMax 四路归并和 32 槽生命周期计划。
- `src/kernel/column_q4.cu`：128 线程，一个 block 处理一个任务的 128 维特征；F256 使用两个 block。版本为 `Q1`、`Sync`、`Bits`、`Float4`、`Pipeline`、`ScalarPipeline`；`Q4Config` 独立选择 H 与 XOR 布局。
- `tests/q4_test.cu`：矩阵逐坐标重建、确定性、回退、tile 覆盖、token/槽位模拟、全部 kernel 对 CPU TF32 参考。
- `bench/benchmark_q4_round2.cu`：固定分组的第二轮消融、容量扫描、原始计时样本及每 SW 统计。
- `bench/benchmark_q4.cu`：Acc、恒等 Q1、分组 Q1、同一计划上的四个 Q4 版本。
- [实现细节](docs/q4-implementation.md)：格式、token、流水线和验证口径。
- [首轮实测](results/q4/report.md)：Products / Arxiv，F128 / F256。

## 构建与验证

沿用项目已配置的 CUDA 12.8 / sm_120 环境。本机 `/tmp` 所在根分区空间不足时，使用项目内临时目录：

```bash
export PATH=/home/shixingguo/miniforge3/envs/gnn/bin:$PATH
mkdir -p build/tmp
export TMPDIR="$PWD/build/tmp"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target q4_test benchmark_q4 benchmark_q4_round2 -j 6
ctest --test-dir build -R '^q4_correctness$' --output-on-failure
/usr/local/cuda-13.0/bin/compute-sanitizer --tool memcheck --error-exitcode 1 ./build/q4_test
/usr/local/cuda-13.0/bin/compute-sanitizer --tool racecheck --error-exitcode 1 ./build/q4_test
```

## 完整数据计时

```bash
bash bench/run_q4_round2.sh 0
# 重跑首轮对照才使用 bash bench/run_q4.sh
```

CSV 记录 T、P、U、全局标量更新数、shared 元素读写数、分组/格式/计划构建时间、回退次数、峰值槽数、包含 Y 清零的完整调用时间、寄存器/shared/local 资源、SM 驻留 block 数及相对 Acc 的全输出误差。使用 CUDA Graph + Events，10 次 warmup，5 次测量，每次 10 个独立完整调用，取中位数；第二轮按固定种子交错各配置，保存全部样本、min/max、波动范围和测量位置。预处理和上传不计入稳态调用。

Q1 每次更新数为 F×P；Q4 为 F×U，shared 读写为 2F(P−U)。`local_bytes` 是 CUDA 报告的 local memory，不能等同于寄存器 spill；spill 查看 `results/q4_round2/build.log` 中的 ptxas 输出。
