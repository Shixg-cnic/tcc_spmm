#!/usr/bin/env python3
import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/q4_datasets'
metadata={x['name'].replace('-','_'):x for x in json.loads((OUT/'datasets.json').read_text())}
summary=[];structure=[]
for dataset in ['ogbn_arxiv','ogbn_products','pubmed','dblp','cora']:
    for f in [128,256]:
        rows={r['method']:r for r in csv.DictReader((OUT/f'{dataset}_f{f}.csv').open())}
        a=rows['Q4_float4_H32'];b=rows['Q4_float4_pipeline_H32']
        t=float(a['complete_ms']);u=float(b['complete_ms'])
        gains=[100*(1-float(b[f'sample{i}_ms'])/float(a[f'sample{i}_ms'])) for i in range(15)]
        summary.append(dict(dataset=dataset,F=f,sync_ms=t,pipeline_ms=u,reduction_pct=100*(1-u/t),acc_ms=float(rows['Acc']['complete_ms']),paired_min_pct=min(gains),paired_max_pct=max(gains),pipeline_wins=sum(v>0 for v in gains),sync_range_pct=a['range_pct'],pipeline_range_pct=b['range_pct'],max_relative_l2=max(float(a['rmse']),float(b['rmse']))))
        if f==128:
            q={r['metric']:r for r in csv.DictReader((OUT/f'{dataset}_f{f}_quantiles.csv').open())}
            meta=metadata[dataset]
            structure.append(dict(dataset=dataset,nodes=meta['rows'],nnz=meta['nnz'],mean_degree=meta['nnz']/meta['rows'],tiles=a['T'],P=a['P'],U=a['U'],saved_updates_pct=100*(1-int(a['U'])/int(a['P'])),cw_p50=q['cw_tiles']['p50'],cw_p95=q['cw_tiles']['p95'],cw_max=q['cw_tiles']['max'],sw_p50=q['sw_tiles']['p50'],sw_p95=q['sw_tiles']['p95'],sw_max=q['sw_tiles']['max'],peak=a['peak_slots']))
for name,rows in [('summary',summary),('structure',structure)]:
    with (OUT/f'{name}.csv').open('w') as file:
        writer=csv.DictWriter(file,fieldnames=rows[0].keys(),lineterminator="\n");writer.writeheader();writer.writerows(rows)
lines=['# Q4 pipeline: five-dataset comparison','', 'RTX 5090 GPU 0, CUDA 12.8.93, sm_120, -O3. Compare `Q4_float4_H32` against `Q4_float4_pipeline_H32`, same grouped format, source mapping, tile order, H32 and output plan. Acc is measured as a same-round reference.','', 'All graphs use deterministic dense random features (test::features, mt19937 seed 17, uniform [-1,1]); F=128/256. This deliberately isolates graph structure from original feature values and is a separate experiment from previous real-feature runs. All times include output clearing and exclude grouping/conversion/upload.','', '15 rounds with a fixed-seed shuffled method order; each sample is a CUDA Graph of 1,000 complete calls for Cora/PubMed/DBLP, 10 for OGB graphs. Warmup is one complete graph per method; reported time is the median of 15 samples. Raw samples and execution order are retained. Positive reduction means pipeline is faster.','', '| Dataset | F | No pipeline ms | Pipeline ms | Time reduction | Faster rounds | Same-round reduction min/max | Acc ms |','|---|---:|---:|---:|---:|---:|---:|---:|']
for r in summary:
    lines.append(f"| {r['dataset']} | {r['F']} | {r['sync_ms']:.6f} | {r['pipeline_ms']:.6f} | {r['reduction_pct']:+.2f}% | {r['pipeline_wins']}/15 | {r['paired_min_pct']:+.2f}% / {r['paired_max_pct']:+.2f}% | {r['acc_ms']:.6f} |")
lines+=['', 'Products F256 has an outlier in the no-pipeline path (range/median about 32%); pipeline range/median is below 1%. Retain this slow sample: the median favors no pipeline, while pipeline is more stable in this run.']
lines+=['','## Graph structure','','| Dataset | Nodes | nnz (directed entries) | Mean degree | CW tiles p50/p95/max | SW tiles p50/p95/max | Output updates saved (1-U/P) | Peak slots |','|---|---:|---:|---:|---|---|---:|---:|']
for r in structure:
    lines.append(f"| {r['dataset']} | {r['nodes']} | {r['nnz']} | {r['mean_degree']:.2f} | {r['cw_p50']}/{r['cw_p95']}/{r['cw_max']} | {r['sw_p50']}/{r['sw_p95']}/{r['sw_max']} | {r['saved_updates_pct']:.2f}% | {r['peak']} |")
wins=sum(r['reduction_pct']>0 for r in summary)
lines+=['', '## Interpretation', '', f'Pipeline has a lower median in {wins}/10 configurations. Small-graph gains show that the benefit is not exclusive to Arxiv. Products must be considered separately; use the table above and retained sample ranges instead of assuming that adding a pipeline always helps.', '', 'The measurements support keeping pipeline as a broadly useful Q4 candidate, with a dataset-dependent choice. They do not establish a universal default, isolate the cause of any slowdown, or prove superiority to other column kernels. Input staging overhead versus output/atomic cost remains a profiling question.']
lines+=['','## Sources and reproducibility','','Cora and PubMed use [Planetoid](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/planetoid.html) graph files; DBLP uses the homogeneous [CitationFull](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/citation_full.html) graph (17,716 papers), not the heterogeneous author-paper DBLP dataset. Added graphs use symmetric union, no self loops, duplicate removal and unit weights. OGB CSR files are reused unchanged. `datasets.json` records source URLs, raw/CSR SHA-256 and shapes; download script checks pinned raw hashes. Raw data remains under ignored data/, not committed.','','```bash','python bench/prepare_q4_datasets.py  # requires NumPy and SciPy','mkdir -p build/tmp','TMPDIR="$PWD/build/tmp" cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=120','TMPDIR="$PWD/build/tmp" cmake --build build --target benchmark_q4_datasets -j4','bash bench/run_q4_datasets.sh','python bench/summarize_q4_datasets.py','```','','Both Q4 methods pass full-output comparison to Acc for all ten dataset/F combinations; Acc is additionally checked against 64 sampled original CSR rows and strided features. No claim of an independent full CPU reference for all outputs. Maximum relative L2 versus Acc: '+f"{max(r['max_relative_l2'] for r in summary):.3e}.",'','Paired min/max compares different sequential executions in the same round, not simultaneous runs or a confidence interval. Graph statistics describe workload differences; they cannot identify hardware bottlenecks without profiling. These results compare the two H32 Q4 variants, not every available column kernel.']
(OUT/'report.md').write_text('\n'.join(lines)+'\n')
for r in summary:print(r['dataset'],r['F'],r['sync_ms'],r['pipeline_ms'],f"{r['reduction_pct']:+.2f}%")
