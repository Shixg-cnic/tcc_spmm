#!/usr/bin/env python3
"""Summarize only the six measured configurations; retain all raw samples."""
import csv
import statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/acc_dequant'
methods=['Acc_original_X','Acc_original_Xhat','Acc_INT4_fused']
rows=[]
validation=[]
for dataset in ['arxiv','products']:
    for f in [128,256,512]:
        prefix=OUT/f'{dataset}_f{f}'
        samples=list(csv.DictReader(Path(str(prefix)+'_samples.csv').open()))
        assert len(samples)==45
        stats={}
        for method in methods:
            ss=[float(r['kernel_ms']) for r in samples if r['method']==method]
            assert len(ss)==15
            stats[method]=(statistics.median(ss),min(ss),max(ss))
        vals=list(csv.DictReader(Path(str(prefix)+'_validation.csv').open()))
        assert len(vals)==9
        meta=next(csv.DictReader(Path(str(prefix)+'_meta.csv').open()))
        bad=sum(int(r['sampled_bad'])+int(r['full_vs_xhat_bad']) for r in vals)
        validation.append((dataset,f,bad,max(float(r['sampled_relative_l2']) for r in vals),max(float(r['sampled_max_abs']) for r in vals),float(meta['quant_output_relative_l2'])))
        rows.append((dataset,f,stats,meta,bad))
with (OUT/'summary.csv').open('w') as stream:
    writer=csv.writer(stream);writer.writerow(['dataset','F','method','median_ms','min_ms','max_ms','validation_bad'])
    for dataset,f,stats,meta,bad in rows:
        for name in methods:writer.writerow([dataset,f,name,*stats[name],bad])
lines=['# 原始 Acc：每个 TC block 内对 X 进行 INT4 反量化', '',
'实验日期：2026-10-20。GPU 0：RTX 5090；CUDA 12.8.93；sm_120，编译选项 `-O3`。', '',
'## 六组实验结果', '',
'只量化 X，A 保持原格式和浮点值。表内时间为 **SpMM kernel 耗时中位数，单位 ms**。变化率 = INT4 耗时 / 原始 Acc 耗时 − 1；负值表示更快。', '',
'| 数据集 | F | 原始 Acc（FP32 X） | Acc + block 内 INT4 反量化 | 耗时变化 | 原始 / INT4 加速比 |',
'|---|---:|---:|---:|---:|---:|']
for dataset,f,stats,meta,bad in rows:
    base=stats[methods[0]][0];fused=stats[methods[2]][0]
    lines.append(f'| ogbn-{dataset} | {f} | {base:.6f} | {fused:.6f} | {(fused/base-1)*100:+.2f}% | {base/fused:.3f}× |')
wins=sum(stats[methods[2]][0]<stats[methods[0]][0] for _,_,stats,_,_ in rows)
lines+=['',f'本次 INT4 融合版本在 {wins}/6 组中的中位耗时低于原始 FP32 X 版本。这里测量的是压缩加载、解包、scale 读取、反量化以及编译后资源变化的**整体影响**，不是单独一次乘法的成本。', '',
'## 在哪里加反量化', '',
'基于 `third_party/Acc-SpMM/include/mma_tf32.cuh` 中最原始的 `tf32_mma_kernel`。使用生成脚本机械复制出 `tf32_mma_kernel_int4`：仅替换 X 参数和 24 处 X 标量加载语句，A 的 bitmap 解码、shared memory 双缓冲、TC block 遍历、MMA、寄存器累加与最终写回保持原代码。原始 kernel 文本和第三方文件不修改。', '',
'修改位置是首块的 `pre loop` 和后续块的 `main loop` 中，填充 `fragB00/01/10/11` 的地方：', '',
'```cpp',
'// 原始：当前 TC block 需要的 X，直接从 global 读浮点数到寄存器',
'fragB00[0] = load_fp32_from_global(d_MatB + sourceIdx0);',
'',
'// INT4：当前 TC block 需要的 X，从 global 读压缩值和 scale',
'fragB00[0] = int4X(d_q, d_scale, feature_dim, sourceIdx0);',
'// int4X: 取所在字节 → 提取低/高 4 bit → 符号扩展 → float(q)*scale[src]',
'```', '',
'辅助函数按线性索引计算源节点和特征位置，然后读取 packed byte 与 `scale[src]`；本次保留这份直接替换实现，没有另加向量化解包或跨 TC block 的反量化缓存。', '',
'反量化得到的 FP32 值直接进入 X 的寄存器片段，再沿用原 `tf32_m16n8k8` 中的 TF32 转换和 MMA。数学上仍计算 `Y=A×X̂`，MMA 内部仍沿用原 kernel 的 `X̂ᵀ×Aᵀ` 排列。计算仍是 TF32 Tensor Core + FP32 累加，**不是 INT4 Tensor Core MMA**。没有新增一个在计时路径里把整张 X 解压为浮点矩阵的 kernel。', '',
'每个稀疏 TC block 为 8 个 dst × 8 个 src；同一个 warp 负责 32 个特征，用两次 M16N8K8 MMA 完成。每个 block 的 X 操作数在加载/预取阶段反量化一次；后续 block 再次引用相同 src 时仍重复反量化。延续原 Acc 流水：加载下一块的 X 时可以先反量化，再计算当前块，不是人为强制在每条 MMA 紧前串行执行。', '',
'本实验保持原 Acc 的启动方式：一个 CTA 负责一个 8 行窗口的全部 F，block=(32,F/32)，因此 F=128/256/512 对应 4/8/16 个 warp，128/256/512 个线程。这与此前 Q4 每 CTA 固定负责 128 维的方式不同。', '',
'## INT4 量化规则', '',
'- 每个源节点的整行 X 使用一个 FP32 scale：`s=max(abs(X[src,:]))/7`；全零行取 `s=1`。',
'- `q=clamp(round_to_nearest_even(X/s),-7,7)`，零点为 0，采用有符号四位补码存储。',
'- 一个字节放两个值：偶数特征在低 4 bit，奇数特征在高 4 bit。反量化为 `X̂=q*s`。',
'- 量化在计时前完成。压缩 X 占 `N*F/2` 字节，另有 `4*N` 字节 scale；原 FP32 X 为 `4*N*F` 字节。', '',
'## 图和特征', '',
'使用此前性能实验的 `data/ogbn_arxiv_full.csr` 和 `data/ogbn_products_full.csr`，本次不改边、不重排节点。它们是已有的双向/自环预处理版本，不是 `dataset/` 的原始边列表。这样保持与此前 Acc 工作负载一致，也避开原始 Acc 不支持空 8 行窗口的问题（原始有向 arxiv 有 631 个空窗口）。', '',
'| 数据集 | 节点数 | 本次 CSR 非零项 | 原始特征维度 |',
'|---|---:|---:|---:|']
for dataset,f,stats,meta,bad in rows:
    if f==128:lines.append(f"| ogbn-{dataset} | {int(meta['nodes']):,} | {int(meta['nnz']):,} | {128 if dataset=='arxiv' else 100} |")
lines+=['',
'X 使用 `data/native_features/ogbn-{arxiv,products}.x.bin` 中的真实节点特征。统一宽度采用 `X_F[src,f]=X_native[src,f % native_F]`：arxiv 128 维保持原值，256/512 重复原特征；products 的 100 维按列循环扩展到 128/256/512。不是随机特征，也不声称 256/512 是数据集原生维度。', '',
'## 公平对照与计时范围', '',
'额外运行 `Acc_original_Xhat`：提前将同一份 INT4 数据反量化成 FP32 X̂，仍调用原始 Acc。这使它与融合版计算相同的 A×X̂，避免把量化后的数值差异误判为实现错误；这一步整矩阵反量化不在任何计时段内。', '',
'| 数据集 | F | 原始 Acc + X̂ ms | 融合 INT4 ms | 相对 X̂ 耗时变化 | 原始 X 范围 ms | 融合 INT4 范围 ms |',
'|---|---:|---:|---:|---:|---:|---:|']
for dataset,f,s,meta,bad in rows:
    a=s[methods[0]];b=s[methods[1]];c=s[methods[2]]
    lines.append(f'| {dataset} | {f} | {b[0]:.6f} | {c[0]:.6f} | {(c[0]/b[0]-1)*100:+.2f}% | {a[1]:.6f}–{a[2]:.6f} | {c[1]:.6f}–{c[2]:.6f} |')
lines+=['',
'每组 15 轮，以固定随机种子 20261020 打乱三种方法的测量顺序；每个样本是 CUDA Graph 内连续 10 次 kernel 的 CUDA Event 时间除以 10，正式计时前每种方法预热 30 次。报告 15 个样本的中位数；范围是实测 min/max，不是置信区间。', '',
'不包含 CPU 格式转换、特征扩展、量化、H2D/D2H、整矩阵预反量化、输出清零。Acc 使用普通 store 覆盖输出，无需每次清零。此前包含清零或不同 X 的表格不能直接与本表作严格性能比较。', '',
'## 正确性与已知限制', '',
'计时前完整检查所有 X 元素的量化编码、scale 和反量化值是否与独立 CPU 计算一致。每种方法执行三次输出校验：抽取 256 个均匀分布的目标行、检查所有 F 列，对照独立 CSR TF32 参考；同时对 X̂ 原始 Acc 和融合版进行全输出比较。逐元素容差为 `0.002+0.001*abs(reference)`。这些检查不等同于完整独立 CSR 输出参考或 GNN 训练精度测试。', '',
'| 数据集 | F | 三次检查不匹配总数 | 最大抽样相对 L2 | 最大抽样绝对误差 | 量化后输出相对 L2（对原始 X，抽样） |',
'|---|---:|---:|---:|---:|---:|']
for dataset,f,bad,rel,mx,qerr in validation:lines.append(f'| {dataset} | {f} | {bad} | {rel:.3e} | {mx:.3e} | {qerr:.3%} |')
lines+=['',
'原始 Acc 曾在之前的实验中被 racecheck 检出首块 X 索引读取与后续覆盖之间缺少同步，见 `../native_features/original_racecheck.log`。按本次“最原始代码”的要求，FP32 和 INT4 两条路径均没有加入屏障；本表不是 `Acc_barrier` 的结果。上述校验只说明本次运行的观察结果，不能证明原始 kernel 没有数据竞争。', '',
'## 复现与文件', '',
'```bash',
'python3 bench/make_acc_int4.py',
'bash bench/run_acc_dequant.sh',
'```', '',
'- `bench/make_acc_int4.py`：机械生成并断言替换 24 个 X 加载位置。',
'- `bench/acc_int4_fused.cu`：原始 Acc 与融合反量化版本。',
'- `bench/benchmark_acc_dequant.cu`：输入准备、正确性校验、交错计时。',
'- `bench/summarize_acc_dequant.py`：从原始 CSV 重建本报告。',
'- 本目录 `*_samples.csv`：全部原始耗时与运行顺序；`*_validation.csv`：三次校验；`*_meta.csv`：图统计和量化误差。',
'- `summary.csv`：三种方法中位数和 min/max；`input_sha256.txt`：原始源码、图和特征校验和；`environment.txt`：硬件/工具链。', '']
(OUT/'report.md').write_text('\n'.join(lines))
print(OUT/'report.md')
