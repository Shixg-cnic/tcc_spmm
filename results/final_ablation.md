# Milestone 5 — Final ablation and cross-milestone comparison

Measured on GPU 0 (RTX 5090, sm_120, driver 580.105.08, CUDA 12.8.93),
full graphs (arxiv 169343x169343, nnz 2 484 941; products 2 449 029 x
2 449 029, nnz 126 167 053), F=128 unless noted, 3 capture groups, median
kernel ms (isolated kernel, same-run Y clear). Acc = the unmodified
`third_party/Acc-SpMM` non-balanced `tf32_mma_kernel` (AdpBME disabled) as
required by the fixed constraints.

## 5.0 Ablation table (F=128, kernel ms median)

| milestone | implementation | arxiv | products | arxiv vs Acc | products vs Acc |
|-----------|----------------|-------|----------|--------------|-----------------|
| M1 | Acc (original, unmodified) | 1.202 | 31.30 | 1.00x | 1.00x |
| M1 | Row (local, bit-tile) | 1.187 | 31.15 | 0.99x | 1.00x |
| M2 | Column C0 (single CW) | 2.577 | 113.76 | 2.14x | 3.63x |
| M2 | Column C1 (cp.async) | 1.988 | 124.09 | 1.65x | 3.96x |
| M2 | Column C3 (SW=16, 32-row cache) | 3.098 | 105.84 | 2.58x | 3.38x |
| M2 | Column C3 (SW=8) | - | 104.04 | - | 3.32x |
| M3 | DynNoCache (dynamic SW, no cache) | 1.115 | - | 0.93x | - |
| M3 | DynCache (dynamic SW, cache) | 1.416 | 103.09 | 1.18x | 3.29x |
| M4 | Int4-DynNoCache (fused INT4) | 1.080 | - | 0.90x | - |
| M4 | Int4-DynCache (fused INT4) | 1.380 | 102.76 | 1.15x | 3.28x |

Best per graph: arxiv Int4-DynNoCache 1.080 ms (0.90x Acc, the only pipeline
variant that beats Acc); products Int4-DynCache 102.76 ms (3.28x Acc — Acc is
irrecoverably ahead on products under the fixed constraints).

## 5.1 What each milestone contributed

- M1: unified engineering + the two local data streams validated against the
  unmodified Acc baseline (Row within 1% of Acc on both graphs).
- M2: Column data stream + C0..C3; identified the structural floor: output
  atomics (the calibrated model attributes ~108.5 ms of the products C0
  113.76 ms at F=128 to atomicAdds). Negative results: C1's cp.async pipeline
  hurts products (124.1 vs 113.8); the SW=16 cache hurts both graphs; the
  stride-132 cache layout gave nothing.
- M3: dynamic SuperWindow scheduling: split the 40 heavy CWs (max 512 tiles)
  killing the 1650-tile serial tail on arxiv (DynNoCache 1.115 vs C0 2.577 =
  0.43x of C0) — the first variant that beats Acc on arxiv (0.93x). On
  products the schedule gains only 2.6% (atomic floor).
- M4: fused INT4 (per-node scale, register dequantisation, packed nibbles):
  4x smaller X stream; arxiv -5.6% over FP32-DynNoCache (1.080), products
  -0.6% (atomic bound); fusion over the independent dequant+FP32 baseline:
  -11% arxiv (2.16 -> 1.92 Int4-C1), -2% products (125.3 -> 122.9).
  Quantisation delta rmse(A.X-hat vs A.X): arxiv 1.78e-2, products 4.12e-2.

## 5.2 Structural bottlenecks (constraint-bound, not tuned away)

1. products output atomics: 2.45M rows x 128 = 1.25 GB Y; every tile does 128
   atomicAdds on Y. Probe-fitted rate 6.72e-9 ms/atomic => ~108 ms floor that
   no scheduling/caching/INT4 change moves (3.28-3.30x Acc is the achievable
   envelope). A full-row write (Row-style) avoids atomics — that is the
   "choose Row/Column" axis the task forbids; Row itself is 31.15 ms (=Acc)
   anyway.
2. arxiv L2-resident Y: 87 MB fits the 5090 L2, so atomics are cheap; the
   kernel is tail/launch bound, fixed by the M3 task table (but the per-SW
   cache machinery still costs more than the 6% atomic saving it removes).
3. C1/C2 double-buffering: negative on products (launch + pipeline overhead
   exceeds the latency hiding), confirmed twice (M2 124.1 vs 113.8; INT4
   Int4-C1 122.9 vs Int4-C0 112.9).
4. INT4 ceiling: only the X stream shrinks; compute (MMA) and output
   (atomics) are unchanged, so INT4 gains are bounded by the X-load share:
   arxiv ~5-6%, products ~0.6%.

## 5.3 Calibrated model final state

Structure t = b.A + a.sum(W_s).FB/P + c.max(W_s).FB + g.H + d.N/P + base.
Probe-fixed b: arxiv 5.12e-9, products 6.72e-9 ms/atomic (graph-specific,
not transferable). FP32 F=128 fit: arxiv within +-22% (tail-dominated),
products +-5% (atomic-dominated); F scaling within +-92% (arxiv) / +-43%
(products) — sub-linear FB over-prediction remains. INT4 refit: F=128 exact
(5 points/5 coefs), same F-scaling caveat. The model is a diagnostic, not a
predictor to be tuned: it quantifies the atomic floor and the per-SW cache
cost, which the ablation above relies on.

## 5.4 Reproducibility

All numbers from the committed CSVs:
results/baseline/* (M1-M2), results/baseline/column_fp32_dynamic.md (M3),
results/int4/* (M4). Benchmarks: benchmark_fp32 / benchmark_int4, groups>=3,
CUDA Graph + external events, warmup 5, spread in min..max columns.
compute-sanitizer memcheck 0 errors / racecheck 0 hazards on all kernels
(row, column, adaptive, int4). ctest 4/4.
