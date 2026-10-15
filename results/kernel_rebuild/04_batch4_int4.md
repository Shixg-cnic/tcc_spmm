# 批次4：最优 FP32 Column kernel → INT4 + Acc Basic INT4 Fused 基线

基线 `7121362` → 批1 `13d54fc` → 批2 `04469bc` → 批3 `3da7b03` → 本批提交见文末。
批4 任务：把最优 FP32 Column kernel（products: V2C-float4；arxiv: V3W-pipe2）转成 INT4
（X 预量化 INT4，kernel 内 fused 反量化），同时实现 Acc Basic INT4 Fused 基线，F=128/256 全测。

## 实现（本批新增）
- `src/kernel/column_fast.cu` / `include/column_fast.cuh`：`fastKernel<WIDE,FTILE,Int4>` 与
  `fastPipeKernel<Stages,WIDE,Int4>` 增加 fused-INT4 X 加载（`fastLoadXInt4`，逐 nibble 反量化，
  与固定 kernel 的 `loadXInt4` 同索引约定）；新增 `launchColumnFastInt4` / `columnFastInt4Attributes`。
  原始 FP32 路径不变（`Int4=false` 默认）。
- `bench/acc_int4_fused.cu`（由 `make_acc_int4.py` 从 third_party 原文件机械生成，**不改动原文件**）：
  (a) 原 Acc-SpMM `tf32_mma_kernel`（FP32 X，原文原样；单 TU 内含转换器 + FP32/INT4 启动器，
  避免与 tests/acc_reference.cu 的 common.hpp 主机辅助函数 ODR 冲突）；
  (b) `tf32_mma_kernel_int4`：同一 kernel 的副本，24 处 X 加载替换为
  `int4X(q,scale,features,idx)`（idx→row/col 除法 + nibble 读取 + 反量化）。
- `bench/benchmark_fast_int4.cu`：同一 CSR/X/F/GPU/stream，6 方法（acc_fp32 / acc_int4_fused /
  fast_v2c_fp32(X-hat) / fast_v2c_int4 / fast_v3w_int4 / c0_int4），CUDA Graph+Events 中位数计时
  （含 Y memset），正确性对照 X-hat 上采样 CPU TF32 参考（require 断言）+ 量化增量（报告）。
- `CMakeLists.txt`：目标 `benchmark_fast_int4`。

## 实测（kernel-only 中位数 ms；方法 = 本文档行名）

| 数据集 | F | acc_fp32 | acc_int4_fused | fast_v2c_fp32 (X-hat) | fast_v2c_int4 | fast_v3w_int4 | c0_int4 |
|---|---|---|---|---|---|---|---|
| arxiv | 128 | 1.248 | 1.899 | 2.207 | 2.159 | **1.677** | 2.576 |
| arxiv | 256 | 1.679 | 2.251 | 2.735 | 2.669 | **2.243** | 3.360 |
| products | 128 | 32.02 | **20.77** | 80.18 | **79.58** | 82.72 | 113.60 |
| products | 256 | 72.34 | **41.22** | 160.17 | **158.82** | 170.79 | 229.71 |

正确性（采样 64 行，对照 X-hat 的 CPU TF32 参考）：全部通过（rmse_xhat=0，与既有
benchmark_int4 的约定一致——输出在采样行上与 double-tf32 参考精确一致）。
量化增量（INT4 的固有误差，报告不断言）：arxiv rmse_x ≈ 0.0178（maxAbs 1.98）、
products ≈ 0.0404（maxAbs 11.45）。

## 结论（代码 + 实测归因）
1. **Column INT4 ≈ Column FP32（−0.8%~−2.8%）**：V2C-INT4 79.58（products F128）/ 2.159（arxiv F128）；
   X 打包省下的读取/流量对 Column 近乎无感——批1 已证其瓶颈是原子 REDG RMW 流（D2c ≈ 22 ms 地板），
   与 X 格式无关。
2. **Acc INT4 Fused 双向分化**：products −35%（20.77 vs 32.02）——Acc 按行重复读 X，X 是 DRAM
   流量大户，打包减半直接省流量；arxiv +52%（1.899 vs 1.248）——X 仅 82 MB L2 常驻，INT4 的
   除法/取模 + 非合并字节读成为纯开销。
3. **INT4 不救 products 目标，比值反而恶化**：Column-INT4 / Acc-INT4 = 3.83×（F128）/ 3.85×（F256），
   而 FP32 比值 2.58× / 2.28×——Acc 的 INT4 收益（35%）远大于 Column 的（0.8%）。
4. 47 ms / 106 ms 目标在 INT4 路径上同样不可达；Column 的瓶颈（跨 CTA 原子 RMW / 贡献者时空分散）
   在本批约束下无解，唯一结构出路仍是批5 动态 SW 协同调度（或放宽图重排约束）。

## 复现
```bash
cd build && make benchmark_fast_int4 -j 16
./benchmark_fast_int4 ../data/ogbn_arxiv_full.csr 128 0 ../data/ogbn_arxiv_full.x.bin
# 重新生成 Acc-INT4 副本（若 third_party 变更）：
/home/shixingguo/miniforge3/envs/gnn/bin/python ../bench/make_acc_int4.py
```
原始 CSV：`results/kernel_rebuild/benchmark_fast_int4_{arxiv,products}_f{128,256}.csv`。
