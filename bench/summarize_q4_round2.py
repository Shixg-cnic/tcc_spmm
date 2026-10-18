#!/usr/bin/env python3
"""Summarize interleaved Q4 experiments; preserve the five raw samples in CSV."""
import csv
import gzip
import shutil
import statistics
from pathlib import Path

root = Path(__file__).resolve().parents[1]
dest = root / 'results/q4_round2'
runs = {}
for d in ('arxiv', 'products'):
    for f in (128, 256):
        with (dest / f'{d}_f{f}.csv').open() as src:
            rows = {r['method']: r for r in csv.DictReader(src)}
        expected = 20 if d == 'arxiv' else 17
        if len(rows) != expected:
            raise RuntimeError(f'Incomplete run: {d} F{f}, got {len(rows)} / {expected}')
        with (dest / f'{d}_f{f}_measurement_order.csv').open() as src:
            order = list(csv.DictReader(src))
        assert len(order) == 5 * len(rows)
        for r in order:
            assert float(r['complete_ms']) == float(rows[r['method']][f'sample{r["round"]}_ms'])
        runs[d, f] = rows

# Archive the large per-CW/SW tables losslessly; keep timing CSVs readable.
checks=[]
for d in ('arxiv','products'):
    previous=None
    for f in (128,256):
        for suffix in ('cw','sw'):
            plain=dest/f'{d}_f{f}_{suffix}.csv'
            if plain.exists():
                with plain.open('rb') as src, plain.with_suffix('.csv.gz').open('wb') as raw:
                    with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as packed:
                        shutil.copyfileobj(src,packed)
                plain.unlink()
        with gzip.open(dest/f'{d}_f{f}_sw.csv.gz','rt') as src:
            sw=[{k:int(v) for k,v in r.items()} for r in csv.DictReader(src)]
        with gzip.open(dest/f'{d}_f{f}_cw.csv.gz','rt') as src:
            cw=[int(r['tiles']) for r in csv.DictReader(src)]
        assert sum(cw)==sum(r['tiles'] for r in sw)==int(runs[d,f]['Q1_grouped']['T'])
        assert sum(r['P'] for r in sw)==int(runs[d,f]['Q1_grouped']['P'])
        assert sum(r['U'] for r in sw)==int(runs[d,f]['Q1_grouped']['U'])
        assert max(r['peak_slots'] for r in sw)==int(runs[d,f]['Q1_grouped']['peak_slots'])
        assert sorted(r['shuffled_position'] for r in sw)==list(range(len(sw)))
        for r in sw:
            pos=r['shuffled_position']
            assert r['task']//256==pos//256
            assert r['tiles'].bit_length()==sw[pos]['tiles'].bit_length()
        if previous is not None:assert previous==sw
        previous=sw
        checks.append(f'{d} F{f}: tasks={len(sw)}, CW={len(cw)}, T/P/U/peak, local-window and bucket permutation checks passed')
    checks.append(f'{d}: all SW statistics and shuffled positions identical across F128/F256')
(dest/'structure_checks.txt').write_text('\n'.join(checks)+'\n')

def val(row, key='complete_ms'):
    return float(row[key])

def best(rows):
    return min((n for n in rows if n.startswith('Q4_')), key=lambda n: val(rows[n]))

lines = ['# Q4 第二轮结果（2026-10-18）', '',
    '固定源列成员、tileOrder 和 rowToken；比较标量双缓冲、H16/H24/H32、A XOR 和局部 task 扰动。'
    'RTX 5090 GPU 0，CUDA 12.8，sm_120；Acc、原序 Q1、分组 Q1 与所有候选在同一轮交错测量。', '',
    '每配置先完整输出核对，再预热 10 次调用。5 轮固定种子打乱测量次序，每个样本为 CUDA Graph 内 '
    '10 次完整调用的平均时间，包含必要的 Y 清零，不包含预处理/上传。以下报告五个样本的中位数；'
    'CSV 保留所有样本、min/max/波动百分比，measurement_order.csv 保留实际位置。', '',
    '## 四组完整调用与验收目标', '',
    '| 数据/F | Acc ms | 原序 Q1 ms | 分组 Q1 ms | 最快 Q4 ms [min,max] | Q4/Acc | 1.10×当轮 Acc | 达标 |',
    '| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |']
for (d, f), rows in runs.items():
    b = rows[best(rows)];a = val(rows['Acc']);t = val(b)
    lines.append(f'| {d}/{f} | {a:.4f} | {val(rows["Q1_identity"]):.4f} | {val(rows["Q1_grouped"]):.4f} | '
                 f'{t:.4f} [{val(b,"min_ms"):.4f}, {val(b,"max_ms"):.4f}] | {t/a:.3f}× | {a*1.1:.4f} | {"是" if t<=a*1.1 else "否"} |')
lines += ['', '最快候选来自本轮样本选择，没有用单次最小时间选优，也不宣称是所有输入上的默认最优：', '']
for key, rows in runs.items():
    lines.append(f'- {key[0]}/F{key[1]}：`{best(rows)}`。')
lines += ['', '固定的上一轮验收线仍可参考：Products F128=34.9808 ms、F256=79.6418 ms；'
          'Arxiv F128=1.3727 ms、F256=1.9668 ms。上表另外使用本轮重新测量的 Acc，避免把不同运行的基线漂移当作优化收益。', '',
          '## 按要求顺序的增量对照（scalar 路径）', '',
          '| 数据/F | 步骤 | 前 ms | 后 ms | 中位数耗时降低 | 同轮样本降低范围 |',
          '| --- | --- | ---: | ---: | ---: | ---: |']
for (d, f), rows in runs.items():
    h = 16 if d=='arxiv' else 24
    pairs = [('A 双缓冲 H32','Q4_bits_H32','Q4_scalar_pipeline_H32'),
             (f'H32→H{h}','Q4_scalar_pipeline_H32',f'Q4_scalar_pipeline_H{h}'),
             (f'A XOR H{h}',f'Q4_scalar_pipeline_H{h}','Q4_scalar_pipeline_auto_xor'),
             ('局部 task 扰动','Q4_scalar_pipeline_auto_xor','Q4_scalar_pipeline_auto_xor_shuffle')]
    for label, a, b in pairs:
        ra, rb = rows[a],rows[b]
        changes = [(1-val(rb,f'sample{i}_ms')/val(ra,f'sample{i}_ms'))*100 for i in range(5)]
        lines.append(f'| {d}/{f} | {label} | {val(ra):.4f} | {val(rb):.4f} | {(1-val(rb)/val(ra))*100:+.2f}% | [{min(changes):+.2f}%, {max(changes):+.2f}%] |')
lines += ['', '“同轮”表示相同测量 round，配置依然顺序执行并且位置不同。范围不是统计置信区间；'
          '微小差异或跨零范围不应解读为稳定收益。容量、XOR 的结论限于表中固定的另一个参数；本轮 XOR 和扰动对照使用最小足够 H。', '',
          '## 全配置计时（ms）', '',
          '| 配置 | Arxiv F128 | Arxiv F256 | Products F128 | Products F256 |', '| --- | ---: | ---: | ---: | ---: |']
names = list(dict.fromkeys(n for rows in runs.values() for n in rows))
for n in names:
    lines.append('| '+n+' | '+' | '.join(f'{val(rows[n]):.4f}' if n in rows else '—' for rows in runs.values())+' |')
lines += ['', '## 实际资源（与 F 无关的 Q4 实例）', '',
          '| 配置 / 数据计划 | H | 寄存器/线程 | shared bytes/CTA | local bytes/线程 | 最大 CTA/SM |',
          '| --- | ---: | ---: | ---: | ---: | ---: |']
seen=set()
for d in ('arxiv','products'):
    for n,r in runs[d,128].items():
        if not n.startswith('Q4_') or n.endswith('_shuffle'):continue
        identity=(n,r['H'])
        if identity in seen:continue
        seen.add(identity)
        lines.append('| '+n+f' / {d} | '+' | '.join(r[k] for k in ('H','registers','shared_bytes','local_bytes','blocks_per_sm'))+' |')
lines += ['', '编译日志的所有新 Q4 实例都是 0 spill stores / loads。实际寄存器占用随 XOR/流水变化，'
          'H16 不保证达到 8 CTA/SM；应读取表中 occupancy API 的实测资源上界。更高的最大驻留数也不保证更短的时间。', '',
          '## 分组、任务结构与预处理', '',
          '| 数据 | T | P | U | peak | CW tiles p50/p95/max | SW tiles p50/p95/max |',
          '| --- | ---: | ---: | ---: | ---: | --- | --- |']
for d in ('arxiv','products'):
    r=runs[d,128]['Q1_grouped']
    with (dest/f'{d}_f128_quantiles.csv').open() as src:q={r['metric']:r for r in csv.DictReader(src)}
    triple=lambda key:'/'.join(q[key][p] for p in ('p50','p95','max'))
    lines.append(f'| {d} | {r["T"]} | {r["P"]} | {r["U"]} | {r["peak_slots"]} | {triple("cw_tiles")} | {triple("sw_tiles")} |')
lines += ['', '每 SW 的 P/U/峰值/扰动位置在 `*_sw.csv.gz`，CW tile 数在 `*_cw.csv.gz`，'
          '大表为无损 gzip CSV，完整分位数在 `*_quantiles.csv`；[结构检查](structure_checks.txt)核对合计、工作量桶及完整排列。扰动保持每个 256-task 局部窗口及各位置的工作量数量级，不改变 T/P/U 或内部生命周期。', '',
          '| 数据/F | 分组/CSC ms | 格式 ms | 计划 ms |', '| --- | ---: | ---: | ---: |']
for (d,f),rows in runs.items():
    r=rows['Q1_grouped'];lines.append(f'| {d}/{f} | '+' | '.join(f'{val(r,k):.2f}' for k in ('group_ms','format_ms','plan_ms'))+' |')
lines += ['', '## 验证与证据', '',
          '- [正确性](correctness.txt)：所有 H/输出/流水/XOR 组合、高槽号 H24/H32、自动分派、容量拒绝和跨局部窗口 task 重排均通过 CPU TF32 参考。',
          '- [memcheck](memcheck.txt) 与 [racecheck](racecheck.txt) 均无错误；[CTest](ctest.txt) 及带实际数据的[旧 fast kernel 回归](legacy_fast_correctness.txt)通过。',
          '- 所有完整数据候选均对 Acc 全输出验证通过；Acc 另对原始 CSR 的抽样行检查。最大 relative RMSE = '+f'{max(val(r,"rmse") for rows in runs.values() for r in rows.values()):.3g}'+ '。',
          '- [SASS](a_layout_sass.txt) 确认 A fragment 是两个 32-bit LDS，异步 staging 是 LDGSTS.E。CPU 枚举证明所测 A 标量地址模式由两路冲突变为无冲突。没有硬件 bank conflict 计数，也未据此宣称全 kernel 无冲突。',
          '- Sync 的只读 PTX 输入已改为 asm 局部临时寄存器；第三方 Acc 保持原样。', '',
          '## 结论', '',
          '1. 标量双缓冲补齐后，Arxiv 相对 Bits H32 降低约 30%/24%，但 Products 两个 F 都增加约 2.8% 耗时；不能预设 F256 标量流水必然受益。', '',
          '2. H16/H24 确实把 shared 降到预期大小，驻留上限从 5 提高到 7/8 CTA，但本轮同版本的较小 H 都慢于 H32。最小足够容量不是最优吞吐策略，保留显式 H32 对照。', '',
          '3. A XOR 的地址/SASS 验证成立。Arxiv 的 Bits/float4 Pipeline 在相同小 H 上有小幅改善，但未抵消容量裁剪的退化；scalar 的改善很小。Products 基本无改善。不能由静态两路冲突推出端到端倍增。', '',
          '4. 本轮局部 task 扰动影响较小，Arxiv scalar 改善约 0.3%–0.4%，Products scalar 退化约 0.4%–0.5%。它没有解释或解决 Arxiv 的 Q1 分组退化，也未接近填补验收差距。', '',
          '5. 新增候选均未超过本轮已有的 H32 float4 路径。四组均未按中位数达到 1.10×当轮 Acc。仍保留显式配置，不将最小 H/XOR/task 扰动推广为默认最优。', '',
          '### 必须保留的波动与跨轮限制', '',
          'Products/F256 的 Q4_float4_H32 五个样本为 '+', '.join(f'{val(runs["products",256]["Q4_float4_H32"],f"sample{i}_ms"):.4f}' for i in range(5))+' ms，跨度约 18.88%。'
          '它只是在这五个样本的中位数上最快，不能称为稳定赢家。较稳的 Q4_float4_pipeline_H32 为 '+f'{val(runs["products",256]["Q4_float4_pipeline_H32"]):.4f}'+' ms，范围为 '+f'[{val(runs["products",256]["Q4_float4_pipeline_H32"],"min_ms"):.4f}, {val(runs["products",256]["Q4_float4_pipeline_H32"],"max_ms"):.4f}]'+'，也未达标。', '',
          '本轮同步 float4 的 Products 时间明显不同于首轮归档；当前构建、模板重构和交错计时组织均与首轮不同，'
          '现有结果不能把该跨轮差值归因于某一项新增优化。原始样本已全部保留，不丢弃慢样本或以最小值代替中位数。', '',
          '下一步不应继续把缩小缓存或改变整个 SW 的执行顺序当作主要提速来源。先定位 float4 路径的波动与输出成本，'
          '再判断是否增加有预算的 SW 成员搜索；当前实验尚不能证明剩余差距主要来自哪一项，也没有证据证明必须更换分组。'
          '本轮不扩展分组算法、不扩大缓存。', '',
          '原始计时：'+', '.join(f'[{d} F{f}]({d}_f{f}.csv)' for d,f in runs)+ '。', '']
(dest/'report.md').write_text('\n'.join(lines))
