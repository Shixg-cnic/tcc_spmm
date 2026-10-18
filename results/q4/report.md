# Q4 首轮结果（2026-10-18）

RTX 5090 / sm_120 / CUDA 12.8。完整调用包含 Y 清零，单位 ms；不含 CPU 预处理和上传。10 次预热，CUDA Graph 每次 10 个独立调用，5 次重放取中位数。详细环境见 [environment.txt](environment.txt)。

## 四组主对照

| 数据集 | F | Acc | Q1 原顺序 | Q1 分组 | 最快 Q4 分组 | Q4 版本 | Q4 / Acc |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| arxiv | 128 | 1.2479 | 2.4127 | 2.8625 | 2.5841 | Q4_pipeline | 2.071× |
| arxiv | 256 | 1.7880 | 3.4049 | 3.5778 | 2.9612 | Q4_pipeline | 1.656× |
| products | 128 | 31.8007 | 120.1199 | 64.2487 | 41.4576 | Q4_pipeline | 1.304× |
| products | 256 | 72.4016 | 236.2784 | 128.8237 | 97.3461 | Q4_bits | 1.345× |

Q1 使用转换外提、同步 A 加载和 scalar atomic；表中 Q4 取本轮最快版本（见版本列），完整消融保留如下。因此主表中 Q1→最终 Q4 的差值不能全部归因于输出归约；更接近同配置的归约对照是 Q1_grouped 与 Q4_bits。

## 同一分组、同一计划的 Q4 消融

| 数据集 | F | Sync | Bits | Float4 | Pipeline |
| --- | ---: | ---: | ---: | ---: | ---: |
| arxiv | 128 | 4.1963 | 4.0821 | 3.8296 | 2.5841 |
| arxiv | 256 | 4.4812 | 4.3598 | 4.0007 | 2.9612 |
| products | 128 | 48.6560 | 48.6387 | 53.8473 | 41.4576 |
| products | 256 | 97.3931 | 97.3461 | 108.2453 | 110.6185 |

## 结构统计

| 数据集 / 分组 | T | P | U | (P−U)/P | 峰值槽数 | 回退组数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| arxiv / Q1_identity | 317,064 | 2,462,547 | 2,414,782 | 1.940% | 7 | 0 |
| arxiv / Q1_grouped | 246,772 | 1,901,544 | 1,764,159 | 7.225% | 14 | 0 |
| products / Q1_identity | 15,896,497 | 126,101,435 | 125,884,570 | 0.172% | 8 | 0 |
| products / Q1_grouped | 9,924,907 | 78,335,815 | 65,197,964 | 16.771% | 20 | 0 |

products：分组使 T 减少 37.57%，P 减少 37.88%；从原始 P 到分组后 U，全局标量更新总减少 48.30%。


全局标量更新：Q1=F×P，Q4=F×U；新增 shared 元素读写=2F(P−U)，逐项数值在 CSV。结构与 F 无关。

## CPU 预处理（毫秒）

| 数据集 | F | 分组/CSC | 格式构建 | Q4 计划 |
| --- | ---: | ---: | ---: | ---: |
| arxiv | 128 | 314.21 | 74.92 | 247.01 |
| arxiv | 256 | 330.59 | 77.49 | 250.48 |
| products | 128 | 22416.87 | 13486.46 | 10948.89 |
| products | 256 | 22320.09 | 15844.58 | 11149.26 |

每个 F 重新构建，用于独立运行；实际应用可跨 F 复用格式和计划。Q1 的 plan_ms 是为本轮结构对照生成 Q4 计划的开销，Q1 kernel 本身不需要计划。

## GPU 资源

| 版本 | 寄存器/线程 | shared bytes/block | local bytes/thread | 最大 block/SM (F128) |
| --- | ---: | ---: | ---: | ---: |
| Acc | 44 | 576 | 0 | 10 |
| Q1_identity | 34 | 256 | 0 | 12 |
| Q1_grouped | 34 | 256 | 0 | 12 |
| Q4_sync | 62 | 17152 | 0 | 5 |
| Q4_bits | 59 | 17152 | 0 | 5 |
| Q4_float4 | 64 | 17152 | 0 | 5 |
| Q4_pipeline | 64 | 17408 | 8 | 5 |

ptxas：所有新 kernel 均为 0 spill stores / 0 spill loads。Pipeline 有 8-byte stack/local，其他新版本为 0。F256 的 Q1/Q4 仍为 128 线程，只增加 grid.y；Acc 为 256 线程，驻留 block/SM 从 10 降为 5。

## 正确性与边界

- [新路径测试](correctness.txt)：精确重建坐标与权重、tile/token/槽位模拟、确定性与预算回退；全部 GPU 版本对 CPU TF32 参考通过。
- [memcheck](memcheck.txt) 0 errors；[racecheck](racecheck.txt) 0 hazards。
- [CTest](ctest.txt) 7/7 通过。旧 column_fast 的无参数 CTest 只显示帮助，另外带实际数据执行了[有效回归](legacy_fast_correctness.txt)，通过。
- 完整 Products、Arxiv 全输出对 Acc，通过项目既有阈值：relative RMSE < 2e-4；max absolute < 2e-4 + max reference × 5e-4。另对 64 个抽样行检查 CPU CSR/TF32 参考。
- 本轮最大 relative RMSE 为 9.98e-06。绝对误差和 RMSE 均保留在原始 CSV，没有按每元素相对误差作保证。
- [Nsight Compute](arxiv_pipeline_ncu.txt) 返回 ERR_NVGPUCTRPERM，没有获得 bank conflict 计数。该文件中的受分析器影响的计时不用于主表。

## 判断

Products 的轻量分组和 Q4 归约均有收益，但首版尚未超过 Acc。Arxiv 的结构改善没有稳定转成相应时间收益；Q4 的缓存、索引及较低驻留度仍需进一步定位。没有寄存器 spill 证据，不能把剩余差距归因于 spill。

float4 和双缓冲均不应视为必然收益：Products 上单独增加 float4 会变慢；F128 的双缓冲进一步改善，但 F256 的 Pipeline 为 110.62 ms，慢于 Bits 的 97.35 ms。不能把 Pipeline 设为所有形状的默认最优版本。下一步应保留这些对照，优先检查 shared 访问、输出指令与任务长度分布；本轮不扩大缓存，也不恢复旧多签名候选池路线。

原始数据：[arxiv F128](arxiv_f128.csv), [arxiv F256](arxiv_f256.csv), [products F128](products_f128.csv), [products F256](products_f256.csv)。
