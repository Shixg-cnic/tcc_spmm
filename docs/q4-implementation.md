# Q4 首版实现说明

本实现以 [用户方案](q4-request.txt) 为范围。旧开发路线停止推进，新入口是 `buildQ4` / `launchQ4`。

## 数据与预处理

`Q4Format` 独立于历史 Column 格式，避免把旧 kernel 中的 `cw*8+k` 假设带到重排路径。`cwCols[8*cw+k]` 始终是原始源列 ID，尾部填 `UINT32_MAX`。CPU 建立结构 CSC，按固定 splitmix64 风格 hash 的最小邻居签名、升序度数、原始 ID 排序，空列放末尾。每 32 列构造一个组，组内以最高度列起始，随后最大化与已覆盖目标行的交集；平局优先高度数、低原始 ID。每组累计邻接超过 262144 则按签名排序后的当前次序每 8 列回退。

行→成员掩码只构建一次。选择一列后，新覆盖的目标行通过成员掩码增量更新其余列的 gain；目标行在同一个 CW 内只计一次。每个 CW 最后按原始源 ID 排序。按 CSR 行顺序散射到各 CW，因此权重流天然与 row-major bitmap 的置位顺序一致，不需要改变数学矩阵。

每最多 8 个严格递增的目标行产生一个 tile；`tileRows` 是原始目标行。格式构建用 32 位索引，溢出或非 canonical CSR 会拒绝。分组/CSC、格式构建、Q4 计划分别计时。

## 执行计划和生命周期

每连续 4 个 CW 构成任务，尾任务可以不足 4 个。CPU 对四个 head tile 的最大有效目标行做稳定归并，相等时选择任务内 CW ID 较小者。生成 `tileOrder`、一字节 `orderCW` 和每行一字节 `rowToken`。

Token 低 5 位存槽号，bit 5 为 HAS_PREV，bit 6 为 HAS_NEXT，bit 7 标记 padding。只出现一次的行不占 shared 槽；首次且有后续的行分配最低空闲槽。最后一次出现的行在当前 tile 结束后释放槽，禁止同 tile 复用。CPU 检查 32 槽容量及任务结束时全部释放。

GPU 仅在最终提交时读取原始目标行；输出 cache 不做整体清零，因为首见行的 MMA C 明确设为零。有前次贡献时直接从 shared 读入 MMA C。每个 tile 结束做 block 同步，下一 tile 才能读取更新后的 C。

## GPU 对照版本

| 版本 | X | 全局输出 | A 加载 |
| --- | --- | --- | --- |
| Q1 | 每 CW 一次加载、一次 TF32 转换 | scalar atomic | 同步 |
| Q4 Sync | SW 开头加载，MMA 内转换 | scalar atomic | 同步 |
| Q4 Bits | SW 开头加载并转换，x0/x1/x2/x3 静态 fragment | scalar atomic | 同步 |
| Q4 Float4 | 同 Bits | warp shuffle 后 float4 atomic | 同步 |
| Q4 Pipeline | 同 Bits | 同 Float4 | 两个 64-float buffer，cp.async |

四个 Q4 版本读取同一份设备格式与任务计划。每 block 128 线程、4 warp，各处理 32 维特征；F256 在 grid.y 分成两个 block。Q4 cache 固定 32×132 float。A fragment 的 TF32 转换在 Bits 及后续版本中供两个 feature part 复用。

Pipeline 在处理当前 tile 前启动下一 A tile 的预取，在当前输出更新后等待复制完成并同步。下一 tile 的 C 不提前加载。float4 分支先由全 warp 完成所有 shuffle，再根据最终行的 token 决定是否 atomic；shared 更新仍按原 MMA 元素所有者普通 load/store。sm_90 以下的编译目标使用标量输出回退，实测 sm_120 使用 float4。

## 验证与解释边界

小矩阵测试逐坐标精确重建权重，独立模拟 tileMax、前后状态及槽所有权，检查每个 tile 恰好执行一次；数值覆盖正负非整数权重、矩形形状、空列、空矩阵、尾 CW/SW、密集复用、预算回退和随机输入。所有版本均与 CPU TF32 参考比较。完整数据使用全输出对 Acc 比较，并对 64 个确定性抽样行的特征检查独立 CSR 参考。

本轮计时用于判断该闭环是否成立，不宣称达到 Acc 性能。尝试 Nsight Compute 时系统返回 `ERR_NVGPUCTRPERM`，未能测量 shared bank conflict 或分离内存/指令瓶颈，因此不能仅凭完整时间给出其因果结论。`Pipeline` 的 8-byte local/stack 不属于 ptxas 报告的寄存器 spill；是否值得去掉需后续对照。
