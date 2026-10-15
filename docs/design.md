# tcc_spmm 设计文档

## 流水线总览（固定约束）

单 RTX 5090（sm_120，CUDA 12.8+）。计算 `Y = A * Dequant(X_int4)`（M4 起），
M3 阶段为 FP32 `Y = A * X` 的完整路径。固定约束：不做全局顶点重排；不混合
Tensor Core 与 CUDA-core；不在 Row/Column 数据流间选择；不做 INT8；不改写
`third_party/Acc-SpMM/`（原样对照，仅非均衡 `tf32_mma_kernel`，禁用 AdpBME）。

固定流水线：ColumnWindow（CW=8 输入列，X 复用/反量化）→ SuperWindow
（SW=16 CW，输出聚合）→ 动态 SW 负载均衡（M3）。

## M3 — 动态 SuperWindow 负载均衡（FP32）

### 格式分层（SW 无关 base + 可变 schedule）

`ColumnBaseFormat`（SW 无关）：CW → 8×8 bitmap tile 的布局
（`colWindowOffset`、`tileLocalBit`、`tileValueOffset`、`values`、
`tileRows`=每 tile 8 个真实行，`UINT32_MAX` 为填充）。行以 tile 本地顺序
存放，任何 schedule 都可直接寻址，无需重编码。

`ColumnSchedule`（CPU 侧）：`SWTask` 表 + 每 SW 的排序唯一行列表、32 行缓存
目录、SW 行范围映射。`SWTask{cwBegin,cwEnd,tileBegin,tileEnd,rowBase,swId,flags}`，
`SWT_SPLIT_CHUNK` 标记重度 CW 拆分出的 chunk 任务。

`ColumnAdaptiveFormat`（GPU 消费）：base + schedule 的显存拷贝，
`tileRowSlot` 按任务用 lower_bound 解析为 SW 局部行槽。

转换流水线：`buildColumnBase`（每矩阵一次）→ `buildAdaptiveSchedule`
（启发式，按配置）→ `buildAdaptiveColumn`（解析 tile 行槽）。

### 确定性启发式分组

代价代理 = CW 的 tile 数。重 CW（> `splitTileThreshold=384`）不参与合并，
按 `splitChunkTiles=256` 切成单 CW chunk 任务；轻 CW 贪心分组，约束
`maxCwsPerSw=64`、`maxTilesPerTask=512`、复用收益 ≥
`minReuseSavingFraction(0.002) × SW 占用`。无随机性，schedule 完全确定。

### 自适应 Kernel

grid.x = 任务数，grid.y = ceil(F/128)。每 CTA 读自己的 `SWTask`，按 CW 钳制
tile 范围，每个 CW 加载一次 X（C2 风格预取）；`UseCache` 模板开关启用
C3 风格 32 行共享缓存（`extern __shared__`，步长 132 避免 bank 冲突）。

### 可校准性能模型

`t = b·A + a·ΣW_s·FB/P + c·max(W_s)·FB + g·H + d·N/P + base`。
b 由无反量化探针固定（arxiv 5.12e-9、products 6.72e-9 ms/atomic，图相关：
arxiv 的 Y 87MB 常驻 L2，products 的 Y 1.25GB 走 DRAM，原子吞吐不同，常数
不可跨图迁移，需按图重校准）。其余系数在 F=128 校准套件（C0–C3 固定 +
3 个动态变体，n=7）上最小二乘拟合。arxiv 由 tail 项主导
（1650-tile 任务序列化 ~2.4ms），products 由原子项主导（108.5ms 占 95%）。

### 实测结论（F=128，kernel ms 中位数）

arxiv：DynNoSplit 2.55 ≈ C0（tail 未被消除）→ 拆分 40 个重 CW 后
DynNoCache 1.12（0.94× Acc，反超原 Acc kernel）。缓存对 arxiv 是净损失
（DynCache 1.42 > DynNoCache 1.12）：每 SW 的缓存机制开销超过 6% 原子节省。
products：原子吞吐地板 ~100ms 主导，DynCache 103.1 为最优 Column
（3.30× Acc），调度只改善 2.6%。负面结果：C1 的 cp.async 流水线在
products 依旧有害（124.5 vs C0 113.6）；DynNoCache 是 products 最差变体
（127.0，293k CTA 启动开销叠加完整原子流量）。

详见 `results/baseline/column_fp32_dynamic.md`。

## M4 — 融合 INT4 反量化（per-node scale，TF32 MMA）

### 数据布局与量化

`DeviceInt4X{q: uint8*, scale: float*}`。量化：每节点块归约 `max|X|`，
`scale = max/7`（全零行 scale=1、q=0），`q = clip(round(x/scale), -7, 7)`；
两个 nibble 打包一个字节（偶特征低 nibble、奇特征高 nibble，奇 F 尾字节补
零），字节索引 `packedByteIndex(node,j,F) = node*((F+1)/2) + j/2`（注意
不是 `(node*(F+1))/2`——除法在乘法之前，奇 F 时两者不同）。

### 融合加载

`columnKernel`/`columnAdaptiveKernel` 增加 `bool Int4` 模板开关，X 参数为
`const void*`：`loadXInt4` 与 `loadX` 同索引、同流水线，每寄存器值做
`nibbleAt(q, packedByteIndex(...)) * scale[col]` 反量化。A 流、MMA 映射、
输出累加与 FP32 路径完全一致（FP32 路径字节级不变，M3 数据不受影响）。
固定 C0/C1/C2/C3 与自适应（cache 开/关）均有 INT4 变体。

### 独立基线（对照）

`dequantizeXInt4` 物化 FP32 X̂（grid-stride 按 pairs 解包），
`Baseline-DeqC1` = 该核 + 原 FP32 C1。融合相对基线的收益就是省掉 X̂ 的
带宽/写入与多一次内核启动。

### 模型重校准（INT4）

3.6 结构不变：原子项 b·A 固定（输出原子与 FP32 相同），其余系数在
INT4 F=128 套件重拟合。拆分 CTA 的重复反量化进入 per-task 项
（`ΣW_s`、`N/P`）：INT4 下 X 重载 4× 更小并带寄存器反量化，对应
arxiv 的 DynCache/DynNoCache 差（1.38 vs 1.08——缓存机制开销依旧为负，
与 FP32 同方向）。

### 实测结论（F=128，kernel ms 中位数）

arxiv：Int4-DynNoCache 1.080（0.94× FP32 DynNoCache 1.144；整体最快），
Int4-C1 1.919（0.95× FP32）；融合较独立反量化基线（Deq+C1 2.161）省
0.24ms。products：Int4-DynCache 102.76 ≈ FP32 DynCache 103.40（0.99×，
原子地板主导，INT4 无法消除）；融合较基线（125.34）省 2.5ms。INT4 量化
delta（A·X̂ vs A·X）rmse：arxiv 1.78e-2、products 4.12e-2，仅报告不优化。

详见 `results/int4/int4_report.md`。

## M5 — 最终消融与结论

各里程碑最佳（F=128，kernel ms 中位数，GPU 0）：

| 里程碑 | 实现 | arxiv | products | arxiv vs Acc | products vs Acc |
|--------|------|-------|----------|--------------|-----------------|
| M1 | Acc（原样） | 1.202 | 31.30 | 1.00× | 1.00× |
| M2 | Column C0/C1/C3 | 2.577/1.988/3.098 | 113.76/124.09/105.84 | 2.14×/1.65×/2.58× | 3.63×/3.96×/3.38× |
| M3 | DynNoCache / DynCache | 1.115 | 103.09 | 0.93× | 3.29× |
| M4 | Int4-DynNoCache / Int4-DynCache | 1.080 | 102.76 | 0.90× | 3.28× |

结论：约束内 Column 流水线在 arxiv 上可反超 Acc（DynNoCache 1.12 → INT4
1.08，0.90×），在 products 上被输出原子地板（~108 ms）锁定在 3.3× Acc
左右——该地板源自 1.25GB Y 的逐 tile 原子累加，调度/缓存/INT4 都无法
移动它（"选 Row/Column 数据流"是任务禁止的轴，且 Row 本身即 Acc 水平）。
负结果均如实记录：C1/C2 双缓冲对 products 有害、SW 缓存机制净开销、
stride-132 无收益、INT4 仅能压缩 X 流（arxiv ~5-6%、products ~0.6%）。
完整消融表与复现说明见 `results/final_ablation.md`。
