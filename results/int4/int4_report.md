# Milestone 4 — Fused INT4 kernel (per-node scale, TF32 MMA)

Measured on GPU 0 (RTX 5090, sm_120, driver 580.105.08, CUDA 12.8.93),
SW=16/CW=8/feature-tile=128, full graphs, F sweep 32..512, 3 capture groups,
median (min..max spread in the CSVs). GPU exclusive during measurement.

## 4.0 Summary

The INT4 path computes `Y = A * Dequant(X_INT4)`: per node `s_i = max|X|/7`,
`q = clip(round(x/s), -7, 7)` (all-zero rows -> s=1, q=0), two q values packed
per byte (even feature -> low nibble, odd -> high; odd F pads). The fused
kernels dequantise in registers while loading X per CW and reuse it across the
CW's tiles, exactly like the FP32 path. All fused kernels are bit-exact
against the TF32 CPU reference on X-hat (rmse 0 in every test).

| graph | best fused INT4 | FP32 counterpart | gain | baseline deq+FP32 | fusion saves |
|-------|-----------------|------------------|------|-------------------|--------------|
| arxiv | Int4-DynNoCache 1.080 ms | 1.144 ms | 0.94x | 2.161 ms | 11% (vs Int4-C1) |
| products | Int4-DynCache 102.761 ms | 103.400 ms | 0.99x | 125.337 ms | 2% (vs Int4-C1) |

INT4 helps where the kernel is sensitive to the X-stream (arxiv: 4x smaller
X loads, register dequantisation, L2-friendly); products is output-atomic
bound (~100 ms floor) so the X savings are second order. The fusion itself is
worth 2-11 % over the independent dequantize-then-FP32 baseline on both graphs.

## 4.1 Quantisation

`quantizeXInt4(const float* x, uint8_t* xq, float* scale, nodes, features, stream)`
— one 128-thread block per node, block reduction for max|X|, per-node scale,
packed stream. `dequantizeXInt4` materialises FP32 X-hat for the independent
baseline. GPU quantisation verified byte-exact against the CPU reference; GPU
dequantisation exact vs CPU X-hat (1e-6) for F=1..512 including odd F.

## 4.2 Fused kernels

`launchColumnInt4SpMM` (fixed C0/C1/C2/C3) and `launchColumnAdaptiveInt4SpMM`
(dynamic, cache on/off) — same kernels as FP32 with an `Int4` template flag
swapping `loadX` for `loadXInt4` (packed-nibble load + `q*s` in registers);
the A stream, MMA mapping and output accumulation are untouched, so the FP32
path is byte-identical to M3.

## 4.3 F=128 results (kernel ms, median)

arxiv (FP32 numbers in the same run):

| method | FP32 | INT4 fused | INT4/FP32 | rmse vs X |
|--------|------|------------|-----------|-----------|
| C0 | 2.566 | 2.483 | 0.97x | 1.78e-02 |
| C1 | 2.022 | 1.919 | 0.95x | 1.78e-02 |
| C3 | 3.102 | 2.970 | 0.96x | 1.78e-02 |
| DynCache | 1.418 | 1.380 | 0.97x | 1.78e-02 |
| DynNoCache | 1.144 | 1.080 | 0.94x | 1.78e-02 |

products:

| method | FP32 | INT4 fused | INT4/FP32 | rmse vs X |
|--------|------|------------|-----------|-----------|
| C0 | 113.590 | 112.946 | 0.99x | 4.12e-02 |
| C1 | 124.310 | 122.881 | 0.99x | 4.12e-02 |
| C3 | 106.280 | 105.687 | 0.99x | 4.12e-02 |
| DynCache | 103.400 | 102.761 | 0.99x | 4.12e-02 |
| DynNoCache | 127.070 | 123.166 | 0.97x | 4.12e-02 |

DequantOnly: arxiv 0.071 ms, products 1.034 ms (independent dequantisation kernel).

## 4.4 All-F sweep (INT4 kernel ms)

arxiv:

| F | C0 | C1 | C3 | DynCache | DynNoCache |
|---|----|----|----|----------|------------|
| 32 | 1.253 | 1.119 | 2.416 | 1.159 | 0.557 |
| 64 | 1.463 | 1.326 | 2.420 | 1.149 | 0.701 |
| 100 | 2.359 | 1.780 | 2.724 | 1.350 | 0.935 |
| 128 | 2.483 | 1.919 | 2.970 | 1.380 | 1.080 |
| 256 | 3.400 | 2.866 | 3.823 | 2.503 | 2.440 |
| 512 | 5.009 | 4.812 | 5.871 | 4.666 | 5.112 |

products:

| F | C0 | C1 | C3 | DynCache | DynNoCache |
|---|----|----|----|----------|------------|
| 32 | 27.654 | 27.636 | 45.176 | 34.345 | 27.207 |
| 64 | 44.655 | 47.498 | 55.777 | 45.628 | 48.300 |
| 100 | 127.402 | 133.042 | 128.100 | 122.620 | 133.824 |
| 128 | 112.946 | 122.881 | 105.687 | 102.761 | 123.166 |
| 256 | 225.722 | 246.373 | 209.763 | 204.833 | 247.368 |
| 512 | 451.699 | 493.558 | 419.600 | 410.039 | 495.848 |

## 4.5 Model recalibration (M4)

The 3.6 model structure transfers unchanged: `t = b*A + a*sum(W_s)*FB/P +
c*max(W_s)*FB + g*H + d*N/P + base` with the probe-fixed atomic rate
(arxiv 5.12e-9, products 6.72e-9 ms/atomic — the output atomics are identical
in INT4). Coefficients re-fit on the INT4 F=128 suite (5 points, see
bench/model_calibration.py int4):

- arxiv: [compute/P, tail, cache/SW, cta/P, base] = [+1.31e-2, -3.45e-4, +3.18e-4, -0.624, +1.1e-4]
  — F=128 fit exact (n=5, 5 coefficients); F-scaling with these coefficients
  over-predicts at F>=256 (+38..+153 %, same sub-linear FB effect as FP32).
- products: [+7.89e-3, -7.66e-3, -1.39e-3, -0.255, +1.4e-6] — F=128 fit within
  4.4%; the atomic term alone (~108.5 ms) explains ~95% of every variant, so
  the scheduling coefficients are second order.

The split CTA re-dequantisation cost is captured by the per-task terms: a split
chunk re-loads its CW's X (4x smaller in INT4) plus the register dequant,
visible as the DynCache/DynNoCache difference on arxiv (1.380 vs 1.080: the
per-SW cache machinery still hurts, exactly like FP32).

## 4.6 Correctness & safety

tests/int4_test.cu: 5 matrices x F in {1,17,32,64,100,128,256,512} x
{C0..C3 fused, adaptive cache on/off, dequant baseline} — all vs the TF32 CPU
reference on X-hat (rmse 0) and the quantization delta reported. Quantize
byte-exact vs CPU; dequant exact. compute-sanitizer memcheck: 0 errors,
racecheck: 0 hazards. ctest 4/4 pass.

## Files

* include/int4_format.cuh, src/format/int4_quantize.cu (new)
* src/kernel/column_spmm.cu, column_adaptive.cu (Int4 template flag)
* tests/int4_test.cu, bench/benchmark_int4.cu (new)
* results/int4/: arxiv_int4_f128.csv, products_int4_f128.csv, arxiv_int4.csv,
  products_int4.csv, int4_memcheck.txt, int4_racecheck.txt, this report
