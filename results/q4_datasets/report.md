# Q4 pipeline: five-dataset comparison

RTX 5090 GPU 0, CUDA 12.8.93, sm_120, -O3. Compare `Q4_float4_H32` against `Q4_float4_pipeline_H32`, same grouped format, source mapping, tile order, H32 and output plan. Acc is measured as a same-round reference.

All graphs use deterministic dense random features (test::features, mt19937 seed 17, uniform [-1,1]); F=128/256. This deliberately isolates graph structure from original feature values and is a separate experiment from previous real-feature runs. All times include output clearing and exclude grouping/conversion/upload.

15 rounds with a fixed-seed shuffled method order; each sample is a CUDA Graph of 1,000 complete calls for Cora/PubMed/DBLP, 10 for OGB graphs. Warmup is one complete graph per method; reported time is the median of 15 samples. Raw samples and execution order are retained. Positive reduction means pipeline is faster.

| Dataset | F | No pipeline ms | Pipeline ms | Time reduction | Faster rounds | Same-round reduction min/max | Acc ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| ogbn_arxiv | 128 | 3.830141 | 2.585318 | +32.50% | 15/15 | +32.42% / +32.59% | 1.247610 |
| ogbn_arxiv | 256 | 3.993322 | 2.958474 | +25.91% | 15/15 | +25.50% / +26.10% | 1.674394 |
| ogbn_products | 128 | 39.744930 | 40.953438 | -3.04% | 0/15 | -3.52% / -2.79% | 31.905148 |
| ogbn_products | 256 | 79.784363 | 82.042938 | -2.83% | 1/15 | -3.60% / +21.30% | 72.458237 |
| pubmed | 128 | 0.083121 | 0.064572 | +22.32% | 15/15 | +22.08% / +22.53% | 0.021791 |
| pubmed | 256 | 0.092963 | 0.082312 | +11.46% | 15/15 | +10.96% / +11.62% | 0.035123 |
| dblp | 128 | 0.078455 | 0.063339 | +19.27% | 15/15 | +19.07% / +19.50% | 0.035393 |
| dblp | 256 | 0.096537 | 0.094042 | +2.58% | 15/15 | +2.36% / +2.90% | 0.046262 |
| cora | 128 | 0.044670 | 0.033525 | +24.95% | 15/15 | +24.52% / +25.29% | 0.010726 |
| cora | 256 | 0.046258 | 0.034276 | +25.90% | 15/15 | +25.59% / +26.13% | 0.012418 |

Products F256 has an outlier in the no-pipeline path (range/median about 32%); pipeline range/median is below 1%. Retain this slow sample: the median favors no pipeline, while pipeline is more stable in this run.

## Graph structure

| Dataset | Nodes | nnz (directed entries) | Mean degree | CW tiles p50/p95/max | SW tiles p50/p95/max | Output updates saved (1-U/P) | Peak slots |
|---|---:|---:|---:|---|---|---:|---:|
| ogbn_arxiv | 169343 | 2484941 | 14.67 | 7/31/2193 | 31/116/2780 | 7.22% | 14 |
| ogbn_products | 2449029 | 126167053 | 51.52 | 19/101/3296 | 89/373/6214 | 16.77% | 20 |
| pubmed | 19717 | 88648 | 4.50 | 2/15/51 | 8/51/92 | 4.70% | 10 |
| dblp | 17716 | 105734 | 5.97 | 3/15/63 | 17/47/89 | 4.03% | 11 |
| cora | 2708 | 10556 | 3.90 | 3/8/31 | 12/22/51 | 3.91% | 7 |

## Interpretation

Pipeline has a lower median in 8/10 configurations. Small-graph gains show that the benefit is not exclusive to Arxiv. Products must be considered separately; use the table above and retained sample ranges instead of assuming that adding a pipeline always helps.

The measurements support keeping pipeline as a broadly useful Q4 candidate, with a dataset-dependent choice. They do not establish a universal default, isolate the cause of any slowdown, or prove superiority to other column kernels. Input staging overhead versus output/atomic cost remains a profiling question.

## Sources and reproducibility

Cora and PubMed use [Planetoid](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/planetoid.html) graph files; DBLP uses the homogeneous [CitationFull](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/citation_full.html) graph (17,716 papers), not the heterogeneous author-paper DBLP dataset. Added graphs use symmetric union, no self loops, duplicate removal and unit weights. OGB CSR files are reused unchanged. `datasets.json` records source URLs, raw/CSR SHA-256 and shapes; download script checks pinned raw hashes. Raw data remains under ignored data/, not committed.

```bash
python bench/prepare_q4_datasets.py  # requires NumPy and SciPy
mkdir -p build/tmp
TMPDIR="$PWD/build/tmp" cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=120
TMPDIR="$PWD/build/tmp" cmake --build build --target benchmark_q4_datasets -j4
bash bench/run_q4_datasets.sh
python bench/summarize_q4_datasets.py
```

Both Q4 methods pass full-output comparison to Acc for all ten dataset/F combinations; Acc is additionally checked against 64 sampled original CSR rows and strided features. No claim of an independent full CPU reference for all outputs. Maximum relative L2 versus Acc: 1.966e-07.

Paired min/max compares different sequential executions in the same round, not simultaneous runs or a confidence interval. Graph statistics describe workload differences; they cannot identify hardware bottlenecks without profiling. These results compare the two H32 Q4 variants, not every available column kernel.
