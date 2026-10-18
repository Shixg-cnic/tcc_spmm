# PubMed：真实 500 维特征，Q4 与 Acc 对比

只测 PubMed / Planetoid：19,717 节点、44,324 无向边 / 88,648 双向条目，原始 500 维 float32 特征。本轮没有测其他真实数据集或原始维度超过 500 的配置。

RTX 5090 GPU 0，CUDA 12.8.93，sm_120，-O3。两个 Q4 版本均 H32、相同分组、tile 顺序和输出生命周期计划。

## 本轮结果

| 方法 | 预打包算子 ms [min,max] | 含 GPU 适配 ms [min,max] |
|---|---:|---:|
| Acc_barrier | 0.068718 [0.068599, 0.069007] | 0.153178 [0.152991, 0.153726] |
| Q4_float4_H32 | 0.135561 [0.135019, 0.135974] | 0.225700 [0.225354, 0.226574] |
| Q4_float4_pipeline_H32 | 0.135572 [0.135025, 0.136262] | 0.214666 [0.213810, 0.215719] |

预打包算子耗时包含输出清零和 SpMM，输入已在 GPU 补零到 512 维。含 GPU 适配耗时包括：连续 500 维 X → 补零到 512 → 清零 + SpMM → 裁回连续 500 维 Y。两方的适配操作一致，不截断、不循环填充、不生成随机替代特征。

每方法/口径预热一个 graph；15 轮固定种子随机顺序交错计时，每个 graph 100 次完整调用，取中位数。两种口径都不包含 CPU 格式转换、分组、H2D/D2H 或参考计算，不是端到端训练时间。

两个 Q4 版本的纯算子耗时基本相同，范围重叠；不能宣称流水在这一形状有显著 kernel 加速。流水版本的包含适配耗时更低，但完整路径差值不能全部归因于预取重叠。两种口径均慢于同步修正版 Acc。

## Acc 原版的竞争问题与修正版

最初使用未修改的 Acc 原版：普通运行的 PubMed 全输出验证通过，但随后 racecheck 在独立小图上报告共享内存读写竞争，并触发 TF32 参考结果不一致。日志保存为 original_racecheck.log / original_racecheck.stdout；原版普通测试、memcheck、计时和顺序也以 original_ 前缀保留。不能把原版当作已经通过竞争检查的稳定基线。

bench/acc_native_barrier.cuh 从原 tf32_mma_kernel 复制，除了函数改名，仅在首次 sparseA2B 索引/特征读取完成后、进入主循环前补一处 __syncthreads()。这阻止快 warp 的预取覆盖仍被慢 warp 读取的初始索引缓冲。原 third_party 文件未修改。上表 Acc_barrier 明确表示这个副本，不是原版。

原版在独立早先一轮测得（仅保留观察值，不与本轮混算）：

- Acc 原版预打包：0.067138 ms；含适配：0.152417 ms。

## 维度和图语义

Q4 入口由仅 128/256 放宽到 128/256/384/512，kernel 计算主体不变。项目现有 Acc 参考入口单次最多 512 维。PubMed 的 500 维仅补 12 个零，输出仍为 500 维。较大原始维度的分块代码没有在本轮验证或测速，不据此作出支持/性能结论。

从 dataset/pubmed/processed/edge_index.npy 构造 A[dst,src]，计算 Y[dst] += X[src]。重复边如存在则相加为权重，保持逐边求和语义；不另加反向边、自环或归一化。inputs.json 保存原始文件 SHA-256 与导出 CSR 统计。

## 验证

本轮三种方法全部真实输出对独立 GPU CSR（TF32 RNA 输入、double 累加）均一致，relative_l2 和 max_abs 为 0。适配小图测试覆盖 F=1/100/128/384/500、非整齐行列、正负非单位权重，并检查逐元素打包、零填充和 CPU 参考。同步修正版 Acc 和 Q4 的 memcheck 为 0 errors，racecheck 为 0 hazards；现有 Q4 回归通过。修正前失败证据未删除。

## 复现

```bash
python bench/prepare_native_features.py --names pubmed
mkdir -p build/tmp
TMPDIR="$PWD/build/tmp" cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=120
TMPDIR="$PWD/build/tmp" cmake --build build --target benchmark_native_features q4_test -j4
ctest --test-dir build -R "^(native_features_correctness|q4_correctness)$" --output-on-failure
build/benchmark_native_features data/native_features/pubmed.csr 500 data/native_features/pubmed.x.bin results/native_features/pubmed_f500
python bench/summarize_native_features.py
```

数据及划分见 [dataset/README.md](../../dataset/README.md)。pubmed_f500.csv 保存全部样本，pubmed_f500_order.csv 保存实际执行顺序。
