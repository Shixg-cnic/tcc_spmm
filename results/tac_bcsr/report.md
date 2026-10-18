# TAC BCSR reproduction results

GPU: RTX 5090 (GPU 0); driver 580.105.08; nvcc 12.8.93; sm_120; -O3.

Original BCSR first-layer kernel body is unchanged. PTX helper compatibility fix is shared with the BCSC reproduction; empty windows use zero tiles and output is padded to 8 rows. See `repro/tac_bcsr/README.md`.

| Matrix | F | Tiles | Kernel ms (no clear needed) | Clear + kernel ms | Relative L2 | Max abs |
|---|---:|---:|---:|---:|---:|---:|
| arxiv | 128 | 317064 | 0.978426 | 1.030269 | 9.891e-06 | 0.141602 |
| arxiv | 256 | 317064 | 1.378387 | 1.531606 | 9.891e-06 | 0.141602 |
| products | 128 | 15896497 | 30.609280 | 31.417078 | 2.579e-06 | 0.432617 |
| products | 256 | 15896497 | 70.260323 | 71.785385 | 2.568e-06 | 0.432617 |

25 small cases passed CPU and GPU CSR validation. All four full-graph cases passed full-output GPU CSR validation (TF32 RNA inputs, double summation). Acceptance thresholds: relative L2 < 2e-4 and max absolute error < 2e-4 + 5e-4 * max absolute reference. FP32 Tensor Core accumulation is not bitwise identical to the double reference. Output was initialized to NaN before the first launch; repeated launches without clearing were also checked.

Compute Sanitizer: memcheck 0 errors; racecheck 0 hazards, 0 errors, 0 warnings. Sanitizers and CTest ran on GPU 1 while full-graph measurements used GPU 0.

Timing: 10 warmups, 5 CUDA Graph samples of 10 calls, median; excludes format conversion and transfers. BCSR overwrites output and does not need a clear. Clear + kernel is retained for comparison of timing scope with BCSC. Prior BCSC results were obtained separately, not through interleaved benchmarking; do not interpret ratios as controlled speedup measurements.
