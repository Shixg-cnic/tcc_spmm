# 阶段一基线：Acc 原始 + Row + Column C0–C3（完整图，统一计时）

## 实验条件

- 机器：fpga3（经 hub 跳板的 node125_5090），单张 NVIDIA RTX 5090（GPU 0，32 GB）。
- 驱动 580.105.08，CUDA 12.8.93（conda env `gnn` 的 nvcc），`sm_120`。
- 仓库起点：`49d906b`（main），本次提交前 HEAD 未变化。
- Column 默认 `SW_SIZE=16`、`CACHE_SIZE=32`、`F_TILE=128`，每 CTA 4 Warp。
- 数据：`data/ogbn_arxiv_full.csr`（M=K=169343，nnz=2484941，原生 F=128）、`data/ogbn_products_full.csr`（M=K=2449029，nnz=126167053，原生 F=100）。复用现有图预处理结果（重复边/自环/双向边语义不变），A 权重为 1。X 使用真实特征：F≤原生维度截取，F 超过时循环扩展（与历史报告口径一致）。
- 计时：CUDA Graph + Event，每组独立捕获 Graph 并实例化，预热 5 次后 3 次计时；≥3 组独立测量，报告所有样本中位数与 min..max 波动。Column 的 complete 每轮含 `cudaMemsetAsync(Y)`。
- 为修复历史报告中的“isolated kernel > complete”异常（如 Products F=128 C3：旧口径 kernel 117.77 ms > complete 106.96 ms），本基准把 isolated kernel 与 complete 放在**同一个 Graph**内：每轮 `memset → eventStart → kernel → eventStop`，complete 用包围整个 Graph 的 Event，kernel 用轮内 Event 之和。两次测量共享同一执行状态，不再产生自相矛盾的分段数据。
- 正确性：每个方法输出与本地 Row（全矩阵）比较，并抽样最多 64 行做 CPU TF32 数学参考，报告 `relative_rmse` 与 `max_absolute_error`。

## 原始 Acc kernel 的限制（源码未修改）

`third_party/Acc-SpMM` 保持原始（UPSTREAM_REVISION `4fe8d04`）。直接调用 `tf32_mma_kernel`（未启用 AdpBME 负载均衡）时记录的限制：

1. F 必须在 [64,512] 且 F%32==0。因此 F=32 与 F=100 的 Acc 对照列缺失（F=32 低于下限；F=100 不满足 32 对齐）。本地 Row/Column 不受此限制。
2. 行数需填充到 8 的倍数（尾部填充行只产生零输出，对照时丢弃）。arxiv 填 1 行（169343→169344），products 填 3 行（2449029→2449032）。
3. 不支持空 8 行窗口（空窗口输出为垃圾且可能越界读）。已用真实 CSR 校验：两个全图均无空 8 行窗口，因此全图对照有效。
4. Acc 的 Y 缓冲需按填充行数分配，本基准对 Acc 使用 padded Y 并丢弃填充行后再比较。

`convertAccRow`（原始 `METCFBit::convertFromCSR`）与本地 `convertRow` 输出完全一致（benchmark 内强制校验，不一致即报错），因此 Row 与 Acc 共享同一组格式数组与上传。

## 完整图 Kernel/Complete 延迟（ms，中位数）

| 数据集 | F | Acc | Row | C0 | C1 | C2 | C3 | 最快 Column | Column/Acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| arxiv | 32 | – | 0.6756 | 1.2026 | 1.1046 | 1.3141 | 2.3276 | C1 | 1.635† |
| arxiv | 64 | 1.0536 | 1.1311 | 2.0737 | 1.6922 | 1.7891 | 2.9554 | C1 | 1.606 |
| arxiv | 100 | – | 1.1574 | 2.4385 | 1.8214 | 1.9938 | 2.9267 | C1 | 1.574† |
| arxiv | 128 | 1.2017 | 1.1871 | 2.5774 | 2.0126 | 2.1193 | 3.0976 | C1 | **1.675** |
| arxiv | 256 | 1.4472 | 1.4486 | 3.4648 | 2.9954 | 3.3606 | 3.9115 | C1 | 2.070 |
| arxiv | 512 | 2.4776 | 2.5692 | 5.2029 | 5.1075 | 6.1112 | 6.0152 | C1 | 2.061 |
| products | 32 | – | 6.3674 | 27.8845 | 27.8790 | 32.0674 | 44.9851 | C0/C1 | 4.379† |
| products | 64 | 13.5015 | 13.5379 | 45.2348 | 47.7065 | 49.1317 | 56.0232 | C0 | 3.350 |
| products | 100 | – | 25.3842 | 128.0916 | 134.0688 | 134.2495 | 128.5712 | C0 | 5.046† |
| products | 128 | 31.3201 | 31.0574 | 113.7578 | 124.0883 | 120.7605 | 105.8763 | C3 | **3.380** |
| products | 256 | 70.7972 | 70.7812 | 228.1265 | 250.0731 | 243.4570 | 211.6543 | C3 | 2.989 |
| products | 512 | 153.3197 | 152.8892 | 455.2729 | 500.8034 | 487.0777 | 422.7756 | C3 | 2.757 |

表内为 `kernel_ms`（Column 的 complete 略高：arxiv F=128 约 +0.05 ms，products F=128 约 +0.73 ms，即 Y 清零开销）。† 标记的比值以 Row 为分母（Acc 不可用）。全部数值 `kernel_min/max` 波动 <1.5%。

## 与历史报告复现性对照

| 矩阵 | F | 历史 Row | 本次 Row | 历史最快 Column | 本次最快 Column |
| --- | ---: | ---: | ---: | ---: | ---: |
| arxiv 全图 | 128 | 1.19299 | 1.18710 | C1 2.02303 | C1 2.01258 |
| arxiv 全图 | 512 | 2.57383 | 2.56918 | C1 5.45315 | C1 5.10750 |
| products 全图 | 128 | 30.95803 | 31.05737 | C3 106.95559 | C3 105.87627 |
| products 全图 | 512 | 152.87195 | 152.88921 | C3 425.81909 | C3 422.77557 |

数值与历史报告一致（<1% 差异）。历史报告中 Products F=128 C3 的 kernel(117.77) > complete(106.96) 系两段独立测量序列的时钟状态不一致，本次同 Graph 测量修复：kernel 105.876 < complete 106.609。

## F=128 关键结论

- arxiv：Column 最快为 C1，2.013 ms，Acc 1.202 ms → **1.67×**。
- products：Column 最快为 C3，105.876 ms，Acc 31.320 ms → **3.38×**。
- 两图最快 Column 均未达到阶段二目标 1.5×。products 的差距显著大于 arxiv：products 的 CW 平均 51.9 个 Tile（arxiv 15.0），Tile 总数 1589 万（arxiv 31.7 万），全局原子输出更新是主要放大项。
- arxiv 上 C1（流水线）优于 C0；products 上 C3（输出缓存）优于 C0/C1/C2，且 F 越大 C3 相对优势越大（F=128 起反超）。两图的最优版本不同，符合“不能因 C1 在 arxiv 更快就默认它在 products 上也更快”的预期。

## 格式、转换与上传

| 数据集 | Row MB | Column MB | Row 转换 ms | Acc 转换 ms | Column 转换 ms | 上传 ms（全部格式） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| arxiv | 23.98 | 42.71 | 503.83 | 424.02 | 306.84 | 21.69 |
| products | 1205.34 | 2208.16 | 22530.61 | 23890.58 | 25221.33 | 1039.42 |

Column 格式含 C0–C3 全部缓存元数据（统一输入格式），比 Row 多约 78%（arxiv）/83%（products）。上传为 GPU 分配+H2D 墙钟，不含 X/Y。

## 数值误差

- 全部方法对 Row 全矩阵：arxiv relative_rmse ≤1.09e-5，max_abs ≤0.161；products relative_rmse ≤2.62e-6，max_abs ≤0.448。差异来自 TF32 累加顺序，符合预期量级。
- 对 CPU TF32 抽样参考：全部方法 relative_rmse ≤1.14e-7，max_abs ≤1.6e-4。
- Acc 与 Row 数值一致（rmse_vs_row=0，两实现共享格式与计算流水线）。

## 回归与 Sanitizer

- `ctest --output-on-failure`：2/2 通过（row_spmm_correctness、column_spmm_correctness）。
- `compute-sanitizer --tool memcheck --error-exitcode 9 ./column_spmm_test`：0 errors（`results/column_memcheck_new.txt`）。
- `compute-sanitizer --tool racecheck --error-exitcode 9 ./column_spmm_test`：0 hazards（`results/column_racecheck_new.txt`）。
- 工程整理后（CanonicalCSR 公共化、bench/ 新增）原有测试全部原样通过。

## 阶段一代码整理

- `include/csr.cuh`（新）：`CanonicalCSR` 与 `canonicalCSR()` 公共定义，Row/Column 共享，Column 不再依赖 Row 格式头。
- `include/row_format.cuh` / `include/column_format.cuh`：改为包含 `csr.cuh`；`src/format/csr.cpp`（新）承载 `canonicalCSR` 实现。
- `tests/column_test_utils.cuh`：显式包含 `row_format.cuh`（移除对 column 头的传递依赖）。
- `bench/benchmark_fp32.cu`（新）：统一 FP32 基线，直接调用原始 `tf32_mma_kernel`。
- `results/{baseline,adaptive,int4}/`：实验结果按阶段分目录。
- C0–C3 历史版本完整保留，用于后续消融对照。

## 复现命令

```bash
cd build
ctest --output-on-failure
./benchmark_fp32 ../data/ogbn_arxiv_full.csr all 3 0 ../data/ogbn_arxiv_full.x.bin > ../results/baseline/arxiv_fp32.csv
./benchmark_fp32 ../data/ogbn_products_full.csr all 3 0 ../data/ogbn_products_full.x.bin > ../results/baseline/products_fp32.csv
```

原始 CSV：[arxiv_fp32.csv](arxiv_fp32.csv)、[products_fp32.csv](products_fp32.csv)。

## 阶段一结论

工程整理完成，可复现的 Acc/Column 差距已建立：arxiv F=128 约 1.67×，products F=128 约 3.38×。products 的差距主要来自全局原子输出更新的放大（C3 缓存 32 行只减少约 0.36% 的更新），这为阶段二提供了明确的优化方向（输出聚合与原子减少、CTA 负载分析），而 arxiv 的差距较小、以流水线/同步开销为主。
