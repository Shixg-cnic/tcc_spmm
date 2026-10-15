# 批次终稿：验收总结（批1–批4，共 4 个提交）

任务书验收标准：不以 kernel 变体数量论成败，而看 Products Column/Acc 比值是否有
“代码 + 实测归因”的实质性下降；目标未达成时，必须留下可验证的瓶颈结论（而非不可验证的
“硬件下限”断言）。本轮按任务书批序执行批1–批4，**批5（动态 SW）本轮未执行**。

## 提交链
- `7121362` 基线 → `13d54fc` 批1（诊断）→ `04469bc` 批2（Fast V1..V4 重建）
  → `3da7b03` 批3（输出规约成本估计 + 方案A 实测判死）→ `7103a1b` 批4（INT4 转换 + Acc INT4 Fused 基线）。
- 每个阶段独立提交；全部负结果保留（批3 两阶段 249.7 ms、批4 INT4 比值恶化均为记录在案的实测负结果）。

## 最终状态（kernel-only 中位数 ms，CUDA Graph + Events，≥3 组，含 Y memset）

FP32 路径（批2 最佳 vs Acc 基线）：
| 配置 | Column 最优 | Acc | 比值 | 目标（≤1.5×） |
|---|---|---|---|---|
| products F=128 | V2C 80.74 | 31.33 | **2.58×** | ✗ |
| products F=256 | V2C 161.27 | 70.77 | **2.28×** | ✗ |
| arxiv F=128 | V3W 1.725 | 1.232 | 1.40× | ✓（1.40×，目标 ≤1.5×） |
| arxiv F=256 | V3W 2.357 | 1.447 | 1.63× | ✗ |

INT4 路径（批4）：Column(V2C-INT4)/Acc(INT4-Fused) = products F128 **3.83×** / F256 **3.85×**；
arxiv F128 0.88× / F256 1.00×。INT4 下 products 比值反而恶化（Acc 的 INT4 收益 35% ≫ Column 的 0.8%）。

**Products 目标未达成**（F128 需 ≤47 ms / 1.5×，F256 需 ≤106 ms / 1.5×）。

## 可验证的瓶颈结论（非“硬件下限”断言；全部有代码 + 测量）
1. **products Column ≈ 98% 输出原子 REDG 流**（批1 D2/D0）：16.14G 次原子累加。
2. **散点因子 ≈ 5.1×**（批1 D2c）：同一 16.1G 原子地址流，L2 常驻+连续布局为 22.06 ms
   （732 G REDG/s），真实流 144 G/s —— 差值 = 跨 CTA (CW,row) 贡献者在 DRAM/Y 上的 RMW 冲突
   + 地址流的不连续；散点与流量均为测得的数值。
3. **指令级重组已到极限**（批2）：V2C-float4 只把 products 从 113.6（C0）→ 80.7 ms（−29%），
   停在 DRAM RMW 写回附近（64.6 GB 读 + 写回 ≈ 72 ms @ ~1.8 TB/s）；arxiv V3W 2.10×→1.40×。
4. **输出规约两方案均死**（批3）：方案A partial 中间量 64.6 GB×2（写分配翻倍流量）实测 249.7 ms
   ≫ V2C 80.7；方案B 行组所有权被 X 复用崩溃（~508 GB 读）与 MMA 8× 膨胀压死（成本模型）。
5. **INT4 不改变 Column 瓶颈**（批4）：X 打包只省 X 流量，而 Column 不是 X 流量受限
   （V2C-INT4 79.58 vs 79.58→80.18 FP32，−0.8%）；Acc 的 INT4 收益（−35%）反而拉大比值。
6. **47/106 ms 目标的唯一结构出路**：批5 动态 SW 协同调度（让同一行的贡献 CW 在执行上相邻，
   把散点原子变回局部规约）或放宽“禁图重排”约束——本轮的固定约束内无第三条可行路径。

## 交付物清单（本轮新增/修改，均提交）
- 批1：`bench/column_output_diagnosis.cu`、`results/kernel_rebuild/01_diagnosis.md` 等。
- 批2：`src/kernel/column_fast.cu`、`include/column_fast.cuh`、`tests/column_fast_test.cu`、
  `results/kernel_rebuild/02_column_fast.md`、fast/fast2 CSVs。
- 批3：`bench/column_partial_reduce.cu`、`results/kernel_rebuild/03_output_reduction.md`、
  partial CSV（旧格式 249.7 合计）。
- 批4：`bench/benchmark_fast_int4.cu`、`bench/acc_int4_fused.cu`（生成器 `bench/make_acc_int4.py`）、
  `results/kernel_rebuild/04_batch4_int4.md`、benchmark_fast_int4_*_f{128,256}.csv（各 6 方法）。
- 全部 5 个 ctest 通过；C0–C3/Dynamic/INT4 既有内核未改动（仅 Fast 系新增 Int4 模板分支）。
