# 批次2：Column Fast 内核（指令级重构）消融报告

基线提交 `7121362`；本批次提交见文末。数据口径与基线一致：CUDA Graph + CUDA Events，
每个配置 3 组 × 重复 5 次取中位数（kernel-only，含 Y memset），GPU 0 独占，同一 CSR / FP32 X / TF32 MMA。
原始数据：`results/kernel_rebuild/fast_{arxiv,products}_f{128,256}.csv`（第一轮 9 变体）与
`fast2_{...}.csv`（扩展组合变体）。

## 1. 交付物与实现

- `include/column_fast.cuh`：`DeviceColumnFast`（直接 tileRows 格式，无 SW slot 间接层）、
  `FastVariant` 枚举、`launchColumnFast` / `columnFastAttributes`。
- `src/kernel/column_fast.cu`：模板内核 `fastKernel<WIDE,FTILE>` + `fastPipeKernel<Stages,WIDE>`。
- `tests/column_fast_test.cu`：正确性（对照 TF32 CPU reference，全图） + `--bench` 消融基准。
- CMake 目标 `column_fast_test`（并入 `libtcc_spmm`）。

变体定义（全部保持约束：每 CTA 一个 CW=8 输入列、X 每特征块只读一次并跨 tile 复用、
8×8 bitmap tile + TF32 MMA、真实稀疏 FP32 权重、不做图重排/协同/混合）：

| 变体 | 机制 |
|---|---|
| V1 | 直接 `tileRows[tile*8+i]` 映射（去掉 C0 的 slot→superWindowRows 间接），标量 atomicAdd |
| V2A | V1 + 地址提升（row*F 预计算）——编译器本已提升，作为独立标签验证 |
| V2B | V1 + warp reshuffle：每 lane 持有同一输出行的连续 2 特征 → `atomicAdd(float2*)`（REDG.64） |
| V2C | V1 + 每 lane 一个输出行 × 4 连续特征 → `atomicAdd(float4*)`（REDG.128） |
| V3 | V1 + cp.async tile 流水（stages=2/4），标量原子 |
| V3W | V3(2 级) + float2 宽原子（流水 × 宽原子组合） |
| V4 | V1 + 内部特征块 FTILE ∈ {32,64,256}（grid.y = F/FTILE，block=FTILE/32×32） |

float4 shuffle 修正记录：宽原子 shuffle 必须对 c0/c1/c2/c3 **分别** `__shfl_sync` 后按目标特征选择
（按源 lane 自己的 group 选寄存器会广播错误值）；V2B/V2C 在 arxiv 全图正确性 rmse ≈ 7e-07 ✓。

## 2. 消融结果（kernel-only 中位数，ms）

### Products（2,449,029 行，nnz 126,167,053）

| F=128 变体 | ms | vs C0 113.57 | F=256 变体 | ms | vs C0 227.53 |
|---|---|---|---|---|---|
| C0（基线） | 113.57 | — | C0 | 227.53 | — |
| V1-direct | 110.18 | −3% | V1 | 220.61 | −3% |
| V2A | 110.15 | −3% | V2A | 220.64 | −3% |
| V2B-f2 | 98.70 | −13% | V2B | 192.28 | −16% |
| **V2C-f4** | **80.74** | **−29%** | **V2C-f4** | **161.27** | **−29%** |
| V2C-ft32 | 88.16 | −22% | V2C-ft32 | 176.86 | −22% |
| V2C-ft64 | 89.60 | −21% | V2C-ft64 | 180.06 | −21% |
| V3-pipe2/4 | 120.3 | +6% | V3-pipe2/4 | 242.0 | +6% |
| V3W-pipe2-f2 | 84.26 | −26% | V3W | 180.06 | −21% |
| V4-ft32 | 111.28 | −2% | V4-ft32 | 223.20 | −2% |
| V4-ft64 | 97.38 | −14% | V4-ft64 | 195.19 | −14% |
| **目标** | **≤47** | | **目标** | **≤106** | |

### Arxiv（169,343 行，nnz 2,484,941）

| F=128 变体 | ms | vs C0 2.584 | vs Acc 1.232 | F=256 变体 | ms | vs C0 3.503 | vs Acc 1.447 |
|---|---|---|---|---|---|---|---|
| V1 | 2.428 | −6% | 1.97× | V1 | 3.314 | −5% | 2.29× |
| V2B-f2 | 2.193 | −15% | 1.78× | V2B | 2.763 | −21% | 1.91× |
| V2C-f4 | 2.212 | −14% | 1.80× | V2C-ft32 | 2.579 | −26% | 1.78× |
| V3-pipe2 | 1.976 | −24% | 1.60× | V3-pipe2 | 2.983 | −15% | 2.06× |
| **V3W-pipe2-f2** | **1.725** | **−33%** | **1.40×** | **V3W-pipe2-f2** | **2.357** | **−33%** | **1.63×** |
| V2C-ft64 | 2.159 | −16% | 1.75× | V2C-ft256 | 3.199 | −9% | 2.21× |

寄存器/占用（128 线程 inst.）：V1/V2A/V2B/V2C/V4 均 56 reg / 288 B smem / 9 CTAs·SM；
V3/V3W 40 reg / 576 B / 12 CTAs·SM；无 spill。V4-ft256 4 CTAs·SM（8 线程块少）最慢。

## 3. 结论与归因（对照批 1 诊断）

1. **Products：输出原子流（REDG RMW + DRAM）仍是绝对瓶颈**。指令级重构合计 −29%
   （113.6 → 80.7，F128；227.5 → 161.3，F256），主要来自 **V2C 宽 float4 原子**
   （REDG.128 把每 warp 的原子指令数降为 1/4，扇区数不变、DRAM RMW 不变，节省的是指令/请求开销）。
   流水（V3）在 Products 反而变慢（+6%）——与批 1 诊断一致：Products 下计算完全被原子掩盖，
   多级 tile 预取只增加共享内存与屏障开销。
2. **Arxiv：计算路径也重要**。V3W（流水+float2）−33%（2.58→1.73 / 3.50→2.36），
   与批 1 的 D1/D0=70%（arxiv 计算占比高）吻合；流水隐藏 tile 解压延迟，宽原子降低 REDG 指令数。
3. **目标状态（诚实声明）**：Products 仍显著慢于 Acc——F=128 为 **2.58×**（80.7 vs 31.3，
   目标 ≤47 需 1.5×），F=256 为 **2.28×**（161.3 vs 70.8，目标 ≤106 需 1.5×）。
   批 1 诊断已量化：16.14G（F128）/ 32.28G（F256）原子被 (CW,row) 结构锁死，
   纯 L2 驻留 REDG 流下限 22 ms（D2c），真实地址流 + DRAM RMW 的缺口是散射与跨 CTA 重取。
   指令级（批 2）已榨尽，**只能由批 3 的输出规约/所有权重构提供结构性下降**。
4. 不实声明：未把任何批 2 结果当作达到目标；所有数字为实测中位数，min/max 见 CSV。

## 4. 复现

```bash
cd build && cmake .. && make column_fast_test -j 16
./column_fast_test ../data/ogbn_arxiv_full.csr 128 0 ../data/ogbn_arxiv_full.x.bin        # 正确性
./column_fast_test ../data/ogbn_products_full.csr 128 0 ../data/ogbn_products_full.x.bin --bench > ../results/kernel_rebuild/fast2_products_f128.csv
```
