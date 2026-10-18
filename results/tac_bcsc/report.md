# TAC BCSC reproduction results

GPU: NVIDIA GeForce RTX 5090 (GPU 0); driver 580.105.08; CUDA compiler 12.8.93; sm_120; -O3.

Kernel: TAC `tf32_computeX128TransposePipe2_BCSC_forward_first_layer`, with a PTX compatibility helper and zero tiles for empty windows. See `repro/tac_bcsc/README.md` for exact scope and commands.

| Matrix | F | Tiles | Kernel ms | Clear + kernel ms | Relative L2 | Max abs |
|---|---:|---:|---:|---:|---:|---:|
| arxiv | 128 | 317064 | 1.439776 | 1.453869 | 6.867e-07 | 0.016113 |
| arxiv | 256 | 317064 | 2.755885 | 2.849802 | 7.236e-07 | 0.016113 |
| products | 128 | 15896497 | 112.782753 | 113.567238 | 1.868e-07 | 0.024414 |
| products | 256 | 15896497 | 205.119583 | 206.597015 | 1.797e-07 | 0.024902 |

Full output validation uses an independent GPU CSR implementation with TF32 RNA rounding and FP64 summation; 25 small cases additionally use a CPU reference. All correctness cases passed; Compute Sanitizer memcheck reports 0 errors and racecheck reports 0 hazards. Sanitizers ran on GPU 1 to avoid interfering with GPU 0 timing.

Times are medians of five CUDA Graph measurements (10 invocations per graph, 10 warmups). CPU format conversion and input transfer are excluded. Pure-kernel repetitions accumulate output; clear + kernel measures independent operations. No speedup claims against earlier experiments are made because they were not measured together.

Original helper compilation failure is preserved in original_compile_failure.log. The imported TAC source remains untouched.
