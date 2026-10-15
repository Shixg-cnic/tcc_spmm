# 批次3：输出规约成本估计 + 方案A（partial+reduce）实测结论

基线 `7121362` → 批1 `13d54fc` → 批2 `04469bc` → 本批提交见文末。
本批按要求：方案A/方案B 成本估计 → 小规模正确性验证 → 只实现最有前景的一个（方案A，全图实测）。

## 1. 成本估计（products F=128；数据来自 CSR 与批1 实测）

中间量（不可压缩）：每个非零元 = 一个 (CW,row) 输出对，`nnz = 126,167,053`；
其部分结果向量为 F 维 → **partial 数据量 = nnz × F × 4B = 64.6 GB**（F=256 为 129.2 GB）。
任何把跨 CTA 累加改成两阶段的方案，都必须把这份中间量写出并读回至少一次。

- **方案A（有界 partial + 逐行规约）**：pass-1 无 RMW 散写 64.6 GB + pass-2 读回 64.6 GB。
  按批1 实测（D1b 散写 1.30 TB/s、DRAM 顺序读 ~1.9 TB/s）估算 ≈ 50 + 34 ≈ 84 ms —— 已超 47 ms 预算。
- **方案B（输出行组所有权）**：tile 的 8 行在图上分散（批1 D3），同一 tile 落入 ≤8 个行组 →
  MMA 膨胀 ≈8×（127M 次、利用率 1/8）、X 复用崩溃（每 (CW,row) 需该 CW 的 X ≈ 508 GB ≫ L2）→ 排除。
- 结构性结论：方案A/B 的中间量都是 64.6 GB×2 ≈ 129 GB；47 ms 预算按 ~1.8 TB/s 只允许 ~85 GB。
  原子路径的 DRAM RMW 同为 64.6 GB 读 + 64.6 GB 写回。**唯一能砍掉中间量的方法 = 让同一行的
  贡献 CW 在执行上相邻（动态 SW 协同调度，批5 保留）或改变图顶点顺序（禁用）。**

## 2. 方案A 实测（`bench/column_partial_reduce.cu`，FB=16 特征块，partial 缓冲 8.07 GB 复用）

pass-1：每 CTA 一个 CW（32 线程），MMA 片段经 per-register shuffle 转 float4 →
**非原子散写**到逐行桶（count[r]/base[r]/ordinal 离线预计算，运行期零原子）；
pass-2：逐行桶求和 → Y。分相计时（CUDA Graph + Events，中位数 ms）。

| 数据集 | F | pass-1 | pass-1 无存储 | pass-2 | 合计 | vs C0 | vs V2C(批2) | vs Acc |
|---|---|---|---|---|---|---|---|---|
| products | 128 | 138.7 | 21.4 | 111.3 | 249.7 | 113.57 | 80.74 | 31.33 |
| arxiv | 128 | 13.8 | 3.87 | 24.2 | 37.8 | 2.584 | 2.212 | 1.232 |

正确性（对照 TF32 CPU reference，全图）：products rmse=1.74e-07、arxiv rmse=8.7e-07 → OK。
（F=256 为同一结构 ×2 特征块数，流量线性翻倍，未单独全测；F=128 结论已决定。）

### 实测归因（代码 + 数字，非“硬件下限”断言）
- pass-1 无存储 = 21.4 ms ≈ 批1 D1（17.4）+ 单 warp CTA 开销 —— 计算/加载路径正常。
- pass-1 存储增量 ≈ 117 ms：64.6 GB 散写 8.07 GB 缓冲（≫ 128 MB L2）→ 每块重复触写同一缓冲，
  **写分配（write-allocate）使 DRAM 流量翻倍到 ~129 GB**，实测 ~550 GB/s 存储速率即 DRAM 上限。
- pass-2 = 111.3 ms：逐线程读自己的行桶（线程=行）→ **未合并读**（580 GB/s，远低于顺序读 ~1.9 TB/s）。
  即使把 pass-2 改 warp-逐行合并读（估计 ~40–60 ms），方案A 合计仍 ≈ 180 ms ≫ V2C 80.7 ms。
- 结论：**方案A 实测比原子 float4 路径差 3.1×（249.7 vs 80.7）**，判死；不继续优化（收益不可达目标）。

## 3. 本批结论与批4 依据
1. 输出规约（方案A/B）在固定约束下均不可行：A 被 partial 中间量 + 写分配翻倍流量压死（实测 249.7 ms），
   B 被 X 复用崩溃与 MMA 膨胀压死（成本估计）。
2. 原子路径的 V2C（float4）仍是 Column 最优：products F128 80.7 ms（3.63×→2.58× Acc）、
   F256 161.3 ms（3.21×→2.28× Acc）；arxiv F128 V3W 1.73 ms（2.10×→1.40× Acc）。
3. 47 ms / 106 ms 目标在“每 CTA 一个 CW + 无图重排 + 无动态 SW + 无协同”约束下无可行路径；
   最终缺口归因：跨 CTA (CW,row) 累加的 DRAM RMW/中间量 64.6 GB×2（F128）是流量级事实，
   唯一出路是批5 动态 SW 协同调度（让同一行的贡献 CW 相邻执行）或放宽图重排约束。
4. 批4 路线：以 V2C（products）/V3W（arxiv）为最优 FP32 kernel 转 INT4 + Acc Basic INT4 Fused 基线。

## 4. 复现
```bash
cd build && make column_partial_reduce -j 16
./column_partial_reduce ../data/ogbn_products_full.csr 128 0 ../data/ogbn_products_full.x.bin --bench   # 正确性 + 分相
./column_partial_reduce ../data/ogbn_products_full.csr 128 0 ../data/ogbn_products_full.x.bin --bench NOSTORE
```
原始 CSV：`results/kernel_rebuild/partial_products_f128.csv`（旧格式合计 249.7）；分相输出见本报告表。
