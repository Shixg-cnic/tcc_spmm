# Q4 第二轮：流水、容量、A 布局与任务顺序

本轮范围以 [审查意见](q4-round2-request.txt) 为准。源列分组、tileMax 顺序、rowToken 编码、FP32 存储 / TF32 MMA / FP32 累加、固定 Q4 和 128 线程都保持原样。没有增加新的成员搜索算法。

## 接口与容量

`Q4Config{version, cacheSlots, xorA}` 独立指定输出/流水配置、容量和 A 布局。新增 `ScalarPipeline` 版本；内核模板的 `Wide` 与 `Pipeline` 是两个独立的布尔参数，覆盖同步/异步 × scalar/float4 四种组合。

`cacheSlots=0` 根据 `DeviceQ4::peakSlots` 选择最小足够容量：≤16 用 H16，≤24 用 H24，否则 H32。也可显式指定 H16/H24/H32 做受控消融。所有启动和属性查询共享同一个模板实例分派函数；容量不足时，在清零/启动之前返回 `cudaErrorInvalidValue`。Q1 不使用 C 缓存。

上传前 `validateQ4Cache` 扫描所有 PREV/NEXT token，检查槽号小于 H 和记录的峰值。`GPUQ4` 在第一笔设备上传前调用该检查。槽号始终通过 `token & 31` 解码，尤其不能用 `H-1`。C stride 保持 132。`q4Attributes(Q4Config, peakSlots, info)` 查询实际 H/XOR/输出/流水实例，而旧的 `Q4Version` 重载保持 H32、线性 A 布局，以便保留首轮对照。

最小足够 H 是内存安全的容量选择，并非“性能必定最好”的策略；本轮实测所有合法 H，报告保留更小 H 的负收益。

## XOR 布局与编译证据

逻辑位置 `t` 的物理地址是 `t ^ ((t >> 3) & 4)`。同步加载、零值填充、cp.async 目标全部使用同一映射；bitmap 位测试和 values/popcount 索引仍使用逻辑 t。读取用 `8*g + (k ^ (g&4))` 和 `8*g + ((k+4) ^ (g&4))`，A 仍为每缓冲 64 float。

CPU 测试枚举 64 个元素，验证双射和 staging 每 warp 的 bank 唯一性；枚举两条 fragment 读取，线性布局每次 16 banks、每 bank 2 个不同地址，XOR 布局每次 32 banks、每 bank 1 个地址。

`results/q4_round2/a_layout_sass.txt` 归档 sm_120 的标量双缓冲 H32 线性/XOR 两个实例。在当前构建中，线性 A 的两个读取是 `LDS R54,[R44]`、`LDS R55,[R44+0x10]`，XOR 是 `LDS R48,[R48]`、`LDS R49,[R49]`；都是两个 32-bit scalar LDS。异步 staging 使用 `LDGSTS.E`。因此上述两条 A 读取的地址枚举适用于实际标量指令形式；不代表 C 缓存无冲突，也不是整个 kernel 的两倍加速保证。硬件 bank conflict 计数仍未获取。

## 局部任务扰动与结构统计

`reorderQ4Tasks` 对连续 256 个 task 分别处理。L=0 单独分桶，L>0 使用 `floor(log2(L))+1`；每桶按固定种子 64-bit hash 排序 task 原始位置，放回该桶原有位置。返回完整的 `Q4Task` 向量；其 begin/end/cw/count 一起移动，不移动或重建 tileOrder、orderCW、rowToken。不同 task 的 F256 两个 feature CTA 映射保持不变。

`Q4Format::taskStats` 与原始 task 顺序对应，保存每 SW 的 P/U/峰值槽数。统计 CSV 记录每 CW tile 数、每 SW 的 tile/P/U/峰值/扰动后位置，以及 0/25/50/75/90/95/99/100 分位数。归档时逐 CW/SW 大表压为无损 `.csv.gz`，计时和分位数保留明文 CSV。此扰动不改变 T/P/U 或算术更新数，只检查完整 task 顺序的敏感性，不能单独解释 Q1 的分组退化。

## PTX 修正与验证

旧 `tf32_m16n8k8` 在 inline asm 内声明局部 `.b32 ta<4>,tb<2>`，将只读输入转换到这些临时寄存器，再送入 MMA，不再改写只读 `%4..%9`。仅修正项目 helper；第三方 Acc 保持原始基线。

测试覆盖所有 H/布局/输出/流水组合和自动 H；额外构造同时占用槽 0/8/16 的 H24 用例，以及槽 24..31 的 H32 用例，防止非二次幂掩码错误。容量不足在 host 验证与 launcher 两端均拒绝。跨 256-task 窗口的扰动测试验证整条记录、工作量桶、局部范围、确定性、tile 生命周期与数值。保留随机正负权重、矩形、空行/列、尾任务和预算回退用例。

## 计时方法

新入口 `benchmark_q4_round2` 同时持有 identity/grouped 格式及一个只替换 task 指针的扰动视图。各候选先作完整输出对 Acc 检查，Acc 另与原始 CSR 的 64 个抽样行核对。每个候选单独捕获 10 次完整调用（含 Y 清零）组成 CUDA Graph。

所有候选预热一次 graph，即每配置 10 次调用。随后做 5 轮，每轮按固定随机种子重新打乱配置测量顺序，在同一 stream 顺序执行。保存每轮的每个样本、实际测量位置、中位数、min/max 和 `(max-min)/median` 百分比。预处理和上传不计入完整调用时间，但单独记录。

主对照包含 Acc、原序 Q1、分组 Q1、各配置中实测最快 Q4；消融分别比较 H32 Bits→标量双缓冲、同一版本 H32→较小 H、同一 H 的线性→XOR、同配置原序→扰动。不要将跨配置复合变化全部归因于其中一项，不把小于样本波动的差值宣称为稳定收益。最优配置来自本轮样本选择，不是跨硬件/输入的普适默认。

复现：

```bash
export PATH=/home/shixingguo/miniforge3/envs/gnn/bin:$PATH
mkdir -p build/tmp
export TMPDIR="$PWD/build/tmp"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target q4_test benchmark_q4_round2 -j 6
ctest --test-dir build -R '^q4_correctness$' --output-on-failure
bash bench/run_q4_round2.sh 0
```
