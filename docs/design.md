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
