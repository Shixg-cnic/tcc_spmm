# CW 执行顺序离线重排实验（experiment/cw-order-only）

- 基线：`main` @ `5ac8989`；本分支只做实验，**不合并 main**。
- 唯一假设：不改数据布局、不改原子数量，仅离线生成 CW 执行顺序（`taskOrder`，`w = taskOrder[blockIdx.x]`），验证能否提高输出 Y 的时间局部性，使 Column V2C 更接近 Acc。
- 本实验**不承诺**达到 Acc 1.5× 目标；结论只回答"重排是否有效、有多大"。

## 1. 代码修改位置
| 文件 | 内容 |
|---|---|
| `include/cw_order.cuh`（新增） | `CWOrderStrategy{Identity,SeededRandom,HotAnchor,BandAnchor}`、`CWOrderConfig`、`CWOrderResult`、`validateCWOrder`、`CWReuseStats`、`CWOrderStructureStats`、`buildCWOrder`、`analyzeCWOrder` |
| `src/format/cw_order.cpp`（新增） | 四策略构建 + 校验 + 离线局部性统计实现 |
| `src/kernel/column_fast.cu`（修改） | `fastKernel` 增加 `template<bool Ordered>` 与 `const U* taskOrder`；`w = Ordered ? taskOrder[blockIdx.x] : blockIdx.x`；新增 `launchColumnFastOrdered`（WIDE=4 / FTILE=128 / FP32）。其余计算（fastLoadX / tf32_m16n8k8 / emitFast / 格式访问）完全不变；`fastPipeKernel` 未动；全部既有调用点显式 `Ordered=false, taskOrder=nullptr` |
| `include/column_fast.cuh`（修改） | 新增 `launchColumnFastOrdered` 声明 |
| `bench/analyze_cw_order.cpp`（新增） | 纯 CPU 离线统计 → structure CSV |
| `bench/benchmark_cw_order.cu`（新增） | 六方法（Acc_Basic / V2C_Original / V2C_IdentityOrder / V2C_RandomOrder / V2C_HotAnchor / V2C_BandAnchor），Kernel-only 与 Complete 双口径，CUDA Graph + Events |
| `tests/cw_order_test.cu`（新增） | 小图 8 情况 × 4 策略 × F128/256 vs CPU TF32 参考；全图模式全量 vs Acc |
| `CMakeLists.txt`（修改） | tcc_spmm 静态库加入 cw_order.cpp；新增 `analyze_cw_order`、`benchmark_cw_order`、`cw_order_test`（注册 ctest `cw_order_correctness`）；未删除任何既有 target |

## 2. 离线算法（复杂度均为 O(有效 (CW,row) 对) + O(W log W)）
- **Identity**：iota。对照组。
- **SeededRandom**：mt19937(20261016) shuffle。对照组，用于判断收益是否只是"打乱调度"带来的。
- **HotAnchor**：行频统计（每个真实行在 tileRows 中出现的次数）→ 每个 CW 取频最高（并列取最小行号）的行作为 anchor → 按 (anchor, w) 排序。
- **BandAnchor**：行频 + 每个 CW 内统计 16384 行主导 band（hist 计数，先命中者胜）→ 按 (dominantBand, anchor, w) 排序。bandRows 只测 8192 / 16384 / 32768。
- 不建 W×W 相似矩阵、无稠密行位图、无 METIS/LP。空 CW 的 anchor=UINT32_MAX 排在最后。
- products W=306,129（order 数组 1,224,516 B）；arxiv W=21,168（84,672 B）。

## 3. 正确性
- `validateCWOrder`：size==W、无越界、无重复、无缺失（四个策略全图均通过）。
- 同 CSR + 同 seed 两次构建结果逐位一致（benchmark 与测试均断言）。
- 小图（19×269，含空 CW、孤立行、重复坐标、跨 CW 行、行数非 8 倍数）：4 策略 × F128/256 全部通过 vs CPU double/TF32 参考（rmse ≈ 1e-9，run-to-run 一致）。
- 全图全量输出 vs Acc：products/arxiv F128 均通过（rmse ≤ 2.6e-6 / 1.0e-5；maxAbs 0.40 / 0.19，均在容差内）。
- compute-sanitizer（小图）：memcheck 0 错误；racecheck 仅报告原始 `tf32_mma_kernel` 的 atomicAdd 读写 hazard（工具不识别原子语义的既有行为），本实验新增 ordered kernel 无任何 hazard。

## 4. 离线结构统计（results/cw_order/*_structure.csv）
products（局部重复率 reuse_ratio = 重复访问 / 总访问，lastSeen+groupId 法）：

| 策略 | G=32 | G=64 | G=128 | 相对 Identity (G=32) |
|---|---|---|---|---|
| Identity | 1.71% | 3.28% | 6.07% | 1.00× |
| SeededRandom | 1.10% | 2.16% | 4.10% | 0.64× |
| **HotAnchor** | **6.29%** | **9.24%** | **13.09%** | **3.68×** |
| BandAnchor 8192 | 4.03% | 5.78% | 8.42% | 2.36× |
| BandAnchor 16384 | 5.01% | 7.11% | 9.94% | 2.93× |
| BandAnchor 32768 | 5.63% | 8.02% | 11.14% | 3.29× |

- 唯一输出行均值（G=128, F=128）：Identity 49,517 / Random 50,558 / Hot 45,817 / Band16384 47,476 —— 各策略相近，说明**重排不改变工作总量**（原子数不变，`num_atomic_updates` 六方法相同）。
- Y 工作字节（G=128, F=256 p95）：Identity 157.1MB / Hot 137.7MB —— 均远超 L2（128MB），所以 HotAnchor 的收益来自把共享行聚集（减少 L2 逐出/原子竞争），而非把 working set 压进 L2。

arxiv：

| 策略 | G=32 | G=64 | G=128 |
|---|---|---|---|
| Identity | 8.62% | 12.19% | 17.41% |
| SeededRandom | 7.75% | 11.77% | 17.29% |
| HotAnchor | 10.20% | 14.67% | ≈19% |
| BandAnchor 16384 | 9.88% | 14.29% | ≈18% |

arxiv 图小（Y 仅 42/84MB，L2 常驻），Identity 的自然 reuse 已高，重排提升有限。

## 5. 性能（results/cw_order/*_f{128,256}.csv；Kernel-only 中位数，ms）
### products（FP32，WIDE=4/FTILE=128 ordered 路径）
| 方法 | F128 kernel | vs Identity | vs Acc ratio | F256 kernel | vs Identity | vs Acc ratio |
|---|---|---|---|---|---|---|
| Acc_Basic | 31.03 | – | 1 | 70.94 | – | 1 |
| V2C_Original | 79.97 | 0.999× | 2.58× | 159.6* | 1.00× | 2.25× |
| V2C_IdentityOrder | 79.87 | 1 | 2.57× | 159.6* | 1 | 2.25× |
| V2C_RandomOrder | 81.39 | **0.98×（慢）** | 2.62× | 162.4 | **0.98×（慢）** | 2.29× |
| **V2C_HotAnchor** | **74.65** | **1.070×** | 2.41× | **148.9** | **1.072×** | 2.10× |
| V2C_BandAnchor 16384 | 77.64 | 1.029× | 2.50× | 155.2 | 1.029× | 2.19× |
| V2C_BandAnchor 8192 | ≈79.4（min） | ≈1.01× | – | – | – | – |
| V2C_BandAnchor 32768 | ≈77.5（min） | ≈1.03× | – | – | – | – |

\* products F256 的 `products_f256.csv` 中 V2C_Original/Identity 中位数出现一次性噪声（186.5ms，min=159.5ms，同轮 Random/Hot 正常）。3 轮复测（bench/rerun_f256.sh，全量重跑）确认真实值稳定：Original 159.65/159.79/186.59（1/3 轮含噪声）、Identity 159.62/159.69/159.64、Random 162.42/162.37/162.39、**HotAnchor 148.86/148.80/148.95（-6.7%）**、Band 155.18/155.21/155.22。噪声来源：每方法开头的 CUDA Graph 重实例化后的前段 sample 偏慢（中位数被抬高），`kernel_min≈159.5` 佐证。报告中采用复测稳定值。

### arxiv
| 方法 | F128 kernel | vs Identity | F256 kernel | vs Identity |
|---|---|---|---|---|
| Acc_Basic | 1.224 | – | 1.618 | – |
| V2C_Original | 2.209 | 1.001× | 2.700 | 1.002× |
| V2C_IdentityOrder | 2.206 | 1 | 2.694 | 1 |
| V2C_RandomOrder | 2.438 | 0.905×（慢） | 2.917 | 0.924×（慢） |
| V2C_HotAnchor | 2.246 | 0.982×（慢） | 2.717 | 0.992×（慢） |
| V2C_BandAnchor 16384 | 2.260 | 0.976×（慢） | 2.730 | 0.987×（慢） |

### 机制开销
V2C_IdentityOrder 与 V2C_Original 逐点几乎一致（±0.1%），证明 Ordered 路径（参数搬运、taskOrder 加载）零额外开销，收益/损失全部来自顺序本身。

## 6. 正 / 负结果
- **正**：HotAnchor 在 products 上稳定快 **6.6%~6.7%**（F128/F256 一致，3 轮复测波动 <0.1%）；BandAnchor（16384/32768）快 2.9%；收益方向与离线 reuse 提升（3.7× / 2.9×）一致。Identity==Original 证明重排机制本身零成本。
- **负**：
  1. 收益远不足以改变大局：products ratio_vs_acc 仅 2.58×→2.41×（HotAnchor F128）—— 距 1.5× 目标差距悬殊；HotAnchor 只是把原子 REDG 竞争略微摊平，无法触及本质瓶颈（16.1G 原子更新 ≈ 2×Acc 原子数，L2-resident 时仍 ~5× 慢于 scatter）。
  2. RandomOrder 在 products 上反而变慢（+1.7%），说明收益**不是**"任意打乱负载均衡"带来的，而是确实的局部性/竞争优化。
  3. arxiv 上所有重排都无益（Hot/Band +1~2% 慢，Random +8~10% 慢）：Y 已 L2 常驻，重排只破坏原有顺序优势。
  4. BandAnchor 随 bandRows 增大单调变好（8192→32768）但都不如 HotAnchor；32768 与 16384 几乎持平。
- **结论**：CW 执行顺序重排是**已验证有效的低成本优化（~7%）**，但属于"治标"：不改变原子数量、不改变 scatter 结构，无法达到 Acc 1.5×。按任务书，此处如实记录为**有限的正面结果（<10%，不足以称为达到目标）**，停止该方向的进一步投入；不合并 main。

## 7. 对后续方向的唯一推断（记录，不执行）
瓶颈依然是输出原子 REDG 流（products C0≈98% 原子流，16.1G 原子更新 ≫ Acc 的 8.7G）。CW 重排的 7% 收益已把"顺序优化"空间基本耗尽；若继续推进 Column，唯一合理路径是改变输出数据流（如新 partial-reduction/row-major 输出）或动态 SW 协同调度 —— 但两者均被本任务书禁止（"不得新增 Partial Reduction / 不得输出 Stripe / 不得动态 SW"）。因此本任务书范围内实验到此为止。

## 8. 复现
```
cd build && cmake .. && TMPDIR=/home/shixingguo/tmp cmake --build . -j 8 --target analyze_cw_order benchmark_cw_order cw_order_test
TMPDIR=/home/shixingguo/tmp ./build/cw_order_test                       # 小图正确性 + sanitizer
TMPDIR=/home/shixingguo/tmp ./build/analyze_cw_order data/ogbn_products_full.csr x > results/cw_order/products_structure.csv
TMPDIR=/home/shixingguo/tmp ./build/benchmark_cw_order data/ogbn_products_full.csr 128 0 data/ogbn_products_full.x.bin 16384
```
（所有运行必须 `TMPDIR=/home/shixingguo/tmp`：`/` 分区满；GPU 0 独占。）
