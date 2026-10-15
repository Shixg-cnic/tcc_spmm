# 01 — 输出路径专项诊断（任务书第一批）

日期：2025-10（本轮基准提交 `7121362` 之后）
环境：RTX 5090 (sm_120)，CUDA 12.8，GPU 0 独占，同一 CSR/FP32 X/TF32 精度。
计时：CUDA Graph + CUDA Event；每组 5 次预热 + 3 次计时（Graph 内 repeat=5）；≥3 组独立测量，报告数值为中位数（min..max）。诊断行均为 kernel-only 时间；D0 为真实 C0 核（另有 64 行采样正确性校验），D1/D1b/D2/D2c 为受控诊断核，**不产生正确 SpMM 结果，不作为性能基线**。

## 1. 分析基线（完整图，重测）

| 数据集 | F | Acc Basic | Row | C0 | C1 | C3 (SW=8) | C3 (SW=16) |
|---|---|---:|---:|---:|---:|---:|---:|
| arxiv | 128 | 1.232 | 1.190 | 2.584 | 2.064 | 2.918 | 3.097 |
| arxiv | 256 | 1.447 | 1.448 | 3.503 | 3.009 | 3.542 | 3.858 |
| products | 128 | 31.334 | 31.408 | 113.570 | 124.204 | 104.520 | 106.024 |
| products | 256 | 70.769 | 70.775 | 227.528 | 250.403 | 208.543 | 212.078 |

单位 ms（纯 kernel 中位数）。原始 CSV：`results/kernel_rebuild/baseline_{arxiv,products}_f{128,256}_sw{8,16}.csv`。
相对 Acc 的比值：products F128 C0=3.63×、C3(SW8)=3.34×；products F256 C0=3.21×、C3(SW8)=2.95×；
arxiv F128 C0=2.10×、C1=1.67×；arxiv F256 C0=2.42×、C1=2.08×。
C3 固定 SW 在 products 上只把 C0 的 3.6× 拉到 3.3×，远未触及 1.5× 目标；C1 双缓冲在 products 上反而更慢（124.2 vs 113.6）。

## 2. 受控诊断（D0–D2c）

`bench/column_output_diagnosis.cu`，SW=16。D1 保留稀疏 Tile 加载 / X 加载 / TF32 MMA / 输出行映射 / Y 地址计算，仅去掉全局写回（每 CTA 一个 checksum 保持地址链存活）；D1b 用普通散写（非原子）写同一地址流；D2 用真实 C0 地址流做相同数量的 `atomicAdd`（常量值，无 MMA 无 X 加载）；D2c 用相同原子数量、全部落在 16 MiB L2 常驻连续 scratch（对照）。

| 诊断 | arxiv F128 | arxiv F256 | products F128 | products F256 |
|---|---:|---:|---:|---:|
| D0 C0（真实） | 2.634 | 3.463 | 113.56 | 227.76 |
| D1 无写回 | 1.858 | 1.972 | 17.38 | 34.03 |
| D1b 非原子散写 | 2.063 | 2.214 | 49.61 | 99.14 |
| D2 原子（真实地址流） | 2.323 | 3.084 | 111.76 | 223.89 |
| D2c 原子（L2 常驻连续） | 0.723 | — | 22.06 | — |

原始 CSV：`results/kernel_rebuild/diag_{arxiv,products}_f{128,256}.csv`、`diag2_{arxiv,products}_f128.csv`。
D0 采样正确性（max-abs vs CPU TF32 参考）：arxiv 4.8e-06、products 3.3e-05~8.4e-05，正常容差内。

关键比率：

- products F128：D2/D0 = **98.4%**；D1/D0 = 15.3%；D0−D2 = 1.8 ms（计算被原子流完全掩盖）。
- products F256：D2/D0 = **98.3%**。
- arxiv F128：D2/D0 = 88.2%；D1/D0 = 70.5%。
- products F128 D2c/D2 = 22.06/111.76 = **5.07×**；arxiv F128 D2c/D2 = 0.723/2.323 = 3.2×。
- 折算有效原子吞吐：真实地址流 16.14G/111.76 ms ≈ **144 G REDG/s**；L2 常驻连续地址流 ≈ **732 G REDG/s**。

## 3. 地址流与竞争分析（D3，离线）

| 量 | arxiv F128 | products F128 |
|---|---:|---:|
| CW 数 | 21,168 | 306,129 |
| Tile 数 | 317,064 | 15,896,497 |
| 有更新的输出行 | 169,343 | 2,449,029 |
| 逻辑原子数 | 315,206,016 | 16,140,983,680 |
| 每行被多少个不同 CW 更新：p50 / p95 / p99 / max / mean | 7 / 40 / 106 / 8,976 / 14.5 | 27 / 164 / 379 / 16,973 / 51.5 |
| 每行 Tile 更新次数 mean / max | 14.5 / 8,976 | 51.5 / 16,973 |
| 更新某行的 CW 索引跨度 mean / max（共 306,129） | 15,065 / 21,167 | 256,491 / 306,122 |
| ≥2 个 CW 更新的行 | 100% | 98.0% |
| ≥16 个 CW 更新的行 | 24.4% | 62.7% |
| Y 工作集 | 82.7 MB（L2 128 MB 内） | 1,195.8 MB（≫ L2） |

结论：

1. **每个输出地址（`Y[row,f]`）平均被 ~51.5 个不同 CTA（不同 CW）更新**（products），即一次输出平均累加 ~51.5 次；arxiv ~14.5 次。
2. **更新同一行的 CW 在 CTA 调度上完全散布**：products 平均跨度 256,491/306,129 —— 一个行的 ~51 个贡献者来自整个网格的不同时段，行缓存行被反复从 DRAM 重取（无时间局部性可利用）；arxiv 跨度也达 15,065/21,168。
3. 特征维度连续：`Y[row*F+f]` 中 f 连续（一个 (CW,row) 的 128 个原子落在 512 B 连续区），但行之间完全分散 → warp 内 8 个原子命中 ~8 个不同 32 B sector。
4. 原子**事务数量**与**地址竞争**两个因素都存在，但数量项（16.14G）与 DRAM 往返项共同主导；竞争本身（同一地址被不同 CTA 同时命中）不是主要序列化来源——吞吐折线（D2c 对照）证明即使无竞争，数量在 L2 常驻下也要 22 ms。

## 4. SASS / 资源分析（cuobjdump，sm_120）

`bench/sass_analysis.sh` 输出保存于本目录运行记录（SASS 全文见构建产物，`cuobjdump -sass column_output_diagnosis`）。

1. **`atomicAdd` 编译为 `REDG.E.ADD.F32.FTZ.RN.STRONG.GPU`**：`red.global` 变体，**无旧值返回**。任务书问题"是否产生不需要的旧值返回"→ **没有**。进一步，PTX `red` 不接受 `.weak` 修饰（`Illegal modifier '.weak' for instruction 'red'`），即 REDG 只有 STRONG 一种作用域形式，不存在通过放宽内存序提升吞吐的空间。
2. C0 核每 Tile 恰好 8 个 REDG 站点（2 part × 4 c 值，循环内复用），C1/C3 同构（C3 9 个）。
3. **地址依赖链**：每 (tile,row) 一次 `LDS`（slot）→ 一次 `LDG`（superWindowRows[rowBase+slot]）→ `IMAD.WIDE`（row×F）+ `LEA`（+feature×4）→ REDG。行值在 8 个 REDG 站点间复用（编译器提升），每 tile 每 warp 仅 ~8–16 次映射 load。**映射元数据不是瓶颈**（见 D1 数值）。
4. **资源**（FP32 变体，`cuobjdump -res-usage`）：C0 = 40 regs / 1312 B shared / 0 spill / 0 stack；C1 = 38 regs / 1600 B；C3 = 53 regs / 1600 B。无寄存器溢出、无 local memory。诊断 D2 仅 19 regs。占用率不受 regs 限制。
5. `cp.async` 预取（C1）在 SASS 中可见（`cp.async` 系列），但 products 上 C1 慢于 C0（124.2 vs 113.6）——与"原子流主导、计算预取不改善输出端"的归因一致。

## 5. 硬件计数器

`ncu` 尝试：`ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters`（目标设备 0）。**计数权限受限，本节不虚构 L1/L2/DRAM/原子吞吐硬件计数**；吞吐数字均来自内核级计时折算（144 G/s 与 732 G/s 为软件口径上限值）。

## 6. 归因结论：3 倍差距中哪条路径最值得优先优化？

**答案：输出端原子 RMW 流本身，且是结构性问题而非指令问题。**

依据（products F128，C0=113.56 ms）：

| 成分 | 时间 | 说明 |
|---|---:|---|
| 计算（MMA+X 加载+映射，D1） | 17.4 ms | 只占 15%；且被原子流完全掩盖（D0−D2 仅 1.8 ms） |
| 非原子散写（D1b−D1） | 32.2 ms | 散写指令/写穿 L2 的代价 |
| 原子 RMW/DRAM 往返（D2−D1b） | 62.1 ms | 同一地址 ~51.5 次跨 CTA 更新 + Y 1.2 GB 的 DRAM RMW |
| 原子总数按 L2 常驻连续计（D2c） | 22.1 ms | 无竞争无 DRAM 的理想原子时间 |

- 原子流（D2）占 C0 的 **98.4%**；其中约 62 ms 来自 DRAM 支持的 RMW 往返（Y 1.2 GB ≫ L2 128 MB，且每行贡献者时间散布 → 行缓存行重取 ~51 次），约 32 ms 来自散写本身，22 ms 是"纯数量"下限。
- **把原子数从 16.14G 降到 0（输出所有权/部分归约）可把 products 从 113.6 ms 拉到接近计算下界 17.4 ms + 单次 Y 写 ~2 ms ≈ 20 ms** —— 目标 47 ms（1.5× Acc）在此结构下可达。
- 把原子数 ÷2–4（宽原子/合并）预计最多到 ~40–60 ms（22 ms 下限 + 散写/RMW 残余），**不足以单独达标**，但作为批次 2 的可测收益保留。
- 指令级微调（V1 行映射直取、V2 重排、V3 流水）不改变原子数量与地址分布，**无法突破 ~112 ms**；V3 流水只对 arxiv 的计算路径（D1=1.86 ms，Acc 的 1.5×）有意义。

**对批次 2/3 的指导：**
1. 批次 2（Fast-V1..V4）在 products 上预期收益有限（测量确认后保留）；在 arxiv 上重点优化计算路径（X 复用/流水/减少行映射），目标接近 Acc（1.20 ms）。
2. **批次 3 是达标的关键路径**：先做方案 A（有界 Partial Results+归约）与方案 B（输出行分组所有权）的成本估算。方案 A 朴素形式（126.1M×(CW,row) 对 × F=128×4B ≈ 64.6 GB partial 写 + 64.6 GB 读 + Y 写 ≈ 130+ GB DRAM 流量）按 ~1.8 TB/s 估算 ≥72 ms，**不可能接近 47 ms，应放弃朴素形式**；方案 B 需核算 X 复用损失（每 (CW,row) 对要求其所有 CTA 加载对应 CW 的 X）与 8×8 Tile 利用率，若退化为 Row 则不作最终方案。
3. 中间形态（批次 3 决策时一并评估）：**CW 批内共享内存/寄存器累计 + 单次写回**（同一 CTA 处理一组 CW，组内重复出现的行先本地累计）——其收益取决于"相邻 CW 共享行的密度"，而 D3 显示行的贡献 CW 在索引空间完全散布（平均跨度 256K），预计组内去重率低，需要实测。

未验证/未实施项：ncu 硬件计数（权限受限）；方案 A/B 的完整实现（批次 3）；任何"已达到硬件下限"的断言（本报告不作出——相反，D2c 对照表明原子路径有 5× 空间，路径是减少/消除原子）。

## 7. 复现命令

```bash
cd build
./benchmark_fp32 ../data/ogbn_{arxiv_full,products_full}.csr {128,256} 3 0 ../data/ogbn_{arxiv_full,products_full}.x.bin {8,16}
./column_output_diagnosis ../data/ogbn_{arxiv_full,products_full}.csr {128,256} 3 0 ../data/ogbn_{arxiv_full,products_full}.x.bin 16
```
