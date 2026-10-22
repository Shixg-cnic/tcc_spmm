#!/usr/bin/env python3
"""Regenerate the 3-dataset x 4-width experiment from raw measurements."""
import csv,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/acc_dequant_ogb'
methods=['Acc_original_X','Acc_original_Xhat','Acc_INT4_fused']
inputs=json.loads((OUT/'inputs.json').read_text());data=[]
for item in inputs:
 name=item['name'].removeprefix('ogbn-')
 for f in [64,128,256,512]:
  prefix=OUT/f'{name}_f{f}';samples=list(csv.DictReader(Path(str(prefix)+'_samples.csv').open()));assert len(samples)==45
  stats={m:tuple(fn([float(r['kernel_ms']) for r in samples if r['method']==m]) for fn in [statistics.median,min,max]) for m in methods}
  checks=list(csv.DictReader(Path(str(prefix)+'_validation.csv').open()));assert len(checks)==9
  bad=sum(int(r['sampled_bad'])+int(r['full_vs_xhat_bad']) for r in checks)
  meta=next(csv.DictReader(Path(str(prefix)+'_meta.csv').open()))
  data.append((name,f,stats,bad,meta,checks))
with (OUT/'summary.csv').open('w') as stream:
 writer=csv.writer(stream,lineterminator='\n');writer.writerow(['dataset','F','method','median_ms','min_ms','max_ms','validation_bad'])
 for name,f,stats,bad,_,_ in data:
  for method in methods:writer.writerow([name,f,method,*stats[method],bad])
lines=['# Acc 每个 TC block 内反量化 X：OGB 扩展实验','',
'范围：ogbn-arxiv、ogbn-products、ogbn-proteins；F=64/128/256/512，共 12 组。按用户后续指示，不测 ogbn-mag；不下载/测试 ogbn-papers100M。','',
'## 性能结果','',
'单位 ms，15 轮中位数。变化率 = INT4 / 原始 Acc − 1；负数表示加速。两者均包含 A 加载、MMA、累加和输出写回；INT4 版本还包含压缩 X 加载、解包、读取 scale 和反量化。量化、上传和 CPU 格式转换不计时。','',
'| 数据集 | F | 原始 Acc（FP32 X） | INT4 融合反量化 | 耗时变化 | 加速比 |','|---|---:|---:|---:|---:|---:|']
for name,f,s,bad,_,_ in data:
 a=s[methods[0]][0];b=s[methods[2]][0];lines.append(f'| ogbn-{name} | {f} | {a:.6f} | {b:.6f} | {(b/a-1)*100:+.2f}% | {a/b:.3f}× |')
lines+=['','## 同数值对照和波动范围','',
'`Acc_original_Xhat` 使用预先完整反量化得到的 FP32 X̂；其数学输入与融合版相同。下表范围是 15 轮 min/max，不是置信区间。','',
'| 数据集 | F | 原始 X 范围 ms | INT4 范围 ms | 原始 Acc + X̂ 中位数 ms |','|---|---:|---:|---:|---:|']
for name,f,s,_,_,_ in data:
 a=s[methods[0]];b=s[methods[2]];lines.append(f'| {name} | {f} | {a[1]:.6f}–{a[2]:.6f} | {b[1]:.6f}–{b[2]:.6f} | {s[methods[1]][0]:.6f} |')
lines+=['','## 数据口径','',
'| 数据集 | 节点数 | 本次 A 的 nnz | 基础 X 维度 |','|---|---:|---:|---:|']
for r in inputs:lines.append(f"| {r['name']} | {r['nodes']:,} | {r['nnz']:,} | {r['native_F']} |")
lines+=['',
'- arxiv/products 复用上次实验的完整 CSR，节点顺序不变，包含此前的对称化/自环预处理。X 使用真实节点特征。',
'- proteins 原始图有 39,561,252 条无向边；A 为双向去重邻接，并添加单位自环，全部非零值为 1。原始 8 维边属性不直接作为 A 权重，而用于构造 X。',
'- proteins 原本没有节点特征：对每个节点的相邻边取 8 维属性均值，作为派生节点特征。它是数据驱动的构造，不是原生节点特征，也不是随机数据。均值按原始边加双向贡献计算，自边贡献两次；孤立节点取零。',
'- 指定宽度使用 `X_F[src,f]=X_base[src,f % base_F]`。F 小于基础宽度时取前 F 列；更大时循环重复。故 arxiv F64 取前 64 列、products F64 取前 64 列，proteins 四种宽度都由 8 维循环扩展。',
'- proteins 重复后的特征相关性很强，这是本次固定宽度输入构造的一部分；结果不代表任意独立随机特征或完整 GNN 训练。',
'',
'来源：[OGB 官方数据集说明](https://ogb.stanford.edu/docs/nodeprop/)；原始下载地址、SHA-256、转换规则见 `inputs.json`。',
'',
'## 实现和计时','',
'沿用原始 Acc 的 `tf32_mma_kernel`，INT4 副本只把 24 处 X 标量加载换成 `int4X`。当前 TC block 的 X 进入寄存器时，提取 4 bit、符号扩展，再执行 `float(q)*scale[src]`；反量化由普通指令完成，结果留在寄存器，不写回 shared/global。之后仍使用 TF32 MMA、FP32 累加，不是 INT4 MMA。',
'',
'每个源节点一个 FP32 scale：`s=max(abs(X))/7`，全零行 s=1；q 按最近偶数舍入后限制在 [-7,7]，两个四位补码值打包成一个字节。压缩 X 加载带来的流量减少与解包计算开销共同决定耗时。',
'',
'本次仅放宽 benchmark 的 F 检查以允许 64；原 Acc 启动器已经支持 64。CTA=(32,F/32)，四种维度分别为 64/128/256/512 线程；A staging 所需的前 64 个线程在 F64 时仍全部存在。',
'',
'每种方法预热 30 次，15 轮固定种子 20261020 打乱方法顺序，每轮每方法测 CUDA Graph 中 10 次 kernel 的平均时间。三个方法共享 A、特征生成规则和输出缓冲；Acc 使用 store 覆盖输出，无需清零。调度器在启动前检查空闲 GPU，并每 2 秒监测其他计算进程；检测到并发时丢弃该组再测。该检测不是驱动级独占锁，不能排除两次轮询间极短的其他任务。实际 GPU 见 `run_state.json`，环境见 `environment.txt`，每组运行前后状态见 `*_gpu_before/after.txt`。本次不进行 Nsight 采集，不需要管理员权限。',
'',
'## 正确性与量化误差','',
'先完整核对 X 的量化编码、scale 和解包结果；每种方法检查三次，独立 CSR 参考覆盖 256 个均匀抽样目标行的全部 F 列，同时将融合输出与 Acc(X̂) 做全输出比较。容差 `0.002+0.001*abs(ref)`。量化误差独立报告，不当作 kernel 实现错误。','',
'| 数据集 | F | 实现不匹配总数 | 最大抽样相对 L2 | 量化输出误差：相对 L2（抽样） |','|---|---:|---:|---:|---:|']
for name,f,_,bad,meta,checks in data:
 lines.append(f"| {name} | {f} | {bad} | {max(float(r['sampled_relative_l2']) for r in checks):.3e} | {float(meta['quant_output_relative_l2']):.3%} |")
lines+=['',
'按要求保留最原始 Acc：此前发现的首块索引同步竞争未修改，见 `../native_features/original_racecheck.log`。检查结果不能证明 kernel 不存在竞争；本实验不等同于训练准确率测试。','',
'## A 矩阵缩略图','',
'![A 矩阵缩略图](figures/ogbn_A_overview.png)','',
'横轴是 src/列，纵轴是 dst/行，节点顺序与实际 benchmark 输入一致。每张矩阵压缩成 768×768 个格子，颜色表示该格内非零项数量，空格为白色；三图使用相同的对数色标。一个格子覆盖的节点数随 N 改变，颜色不是归一化边密度。','',
'参考用户提供的 `/home/shixingguo/work/GNN/unitls/pic_heatmap.py`：保留按位置累加 nnz 的方法、配色与最近邻显示；改为分块 CSR + bincount，加速大图聚合，不需要先转 MTX。所有格子计数之和均检查等于 CSR nnz；统计保存在 `figures/metadata.json`。','']
for r in inputs:lines.append(f"- [{r['name']} PNG](figures/{r['name']}.png) / [PDF](figures/{r['name']}.pdf)")
lines+=['','## 复现','',
'```bash',
'PY=/home/shixingguo/miniforge3/envs/gnn/bin/python',
'$PY bench/prepare_acc_ogb.py',
'TMPDIR="$PWD/build/tmp" cmake --build build --target benchmark_acc_dequant -j4',
'python3 bench/wait_run_acc_ogb.py  # 等待无其他计算进程的 GPU，完成后自动汇总',
'PYTHONPATH="$PWD/build/plotdeps" $PY bench/plot_acc_ogb_heatmaps.py',
'python3 bench/summarize_acc_ogb.py','```','',
'完整原始时间与测量顺序：`*_samples.csv`；三次检查：`*_validation.csv`；概要：`summary.csv`。','']
(OUT/'report.md').write_text('\n'.join(lines));print(OUT/'report.md')
