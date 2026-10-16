# FP32 Column versus Acc — independent output-stripe experiment

Branch: `experiment/column-output-stripes-fp32`; base: `5ac8989`.

**Status: experimental, NOT benchmarked on RTX 5090, NOT proven faster or correct on GPU.**
The execution environment used to prepare this branch has no CUDA compiler or GPU;
all performance figures below are targets or previously reported reference figures.

## Scope and design

This branch adds an alternative FP32 Column kernel without changing original
Row, Acc, C0–C3, Dynamic, or INT4 code. F is restricted to **128/256**.
The original base format is a CW=8 input-window stream with 8x8 bitmap tiles.
Instead of traversing the entire CW's output rows across the 1.2GB Products Y
buffer, the converter groups the contributions by a *contiguous output-row stripe*
without renumbering or reordering graph vertices.

For each stripe, the host builds a separate CW→tile stream. Each GPU launch
computes **one output stripe** and the host launches stripes sequentially in
the same CUDA stream, so any current-stripe Y range is likely to remain in
L2 while its incoming CW contributions are processed. An input X fragment
is still read once *per CW per stripe* and reused across that CW's tiles.
The compute uses the same TF32 m16n8k8 primitive and actual FP32 A weights
as existing ColumnFast. There are scalar and float4 atomic-output variants.

This is a cache-working-set experiment, not a variant of adjacent-CW SW
aggregation: it deliberately changes **temporal output locality**, not just
the number of updates. It does **not** reduce the number of (CW,row)
contributions. It may multiply CW loads and CTA count by the number of
stripes, increase bitmap tile fragmentation, and incur extra kernel launches.
These costs must be measured before calling the approach viable.

## Build and run on the RTX 5090 node

```bash
git fetch origin
git switch experiment/column-output-stripes-fp32
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 16 --target column_striped_test
./build/column_striped_test data/ogbn_products_full.csr 128 65536 0 data/ogbn_products_full.x.bin
./build/column_striped_test data/ogbn_products_full.csr 256 32768 0 data/ogbn_products_full.x.bin
./build/column_striped_test data/ogbn_arxiv_full.csr 128 65536 0 data/ogbn_arxiv_full.x.bin
./build/column_striped_test data/ogbn_arxiv_full.csr 256 32768 0 data/ogbn_arxiv_full.x.bin
```

The program compares results against an unmodified Acc Basic FP32 kernel
and reports complete SpMM latency including Y memset. For a sensitivity
study use row-stripe widths `16384,32768,65536,131072,262144`, initially
on Products F=128. In F=128, a 65,536-row stripe has a 32 MiB Y working
set; in F=256, a 32,768-row stripe has a 32 MiB working set.

The converter intentionally makes repeated offline scans of source tiles
to reduce complexity of the experimental implementation. Conversion time
is NOT included in the kernel benchmark. Measure conversion time and format
memory cost separately before adopting the layout in production.

## Stop conditions

Do not replace the production Column kernel unless:
1. `ColumnStripedScalar` and `ColumnStripedFloat4` pass full-output
   comparisons against original Acc and a sampled TF32 CPU reference;
2. compute-sanitizer memcheck/racecheck pass on test matrices including
   weighted, irregular, rectangular, empty-window and long-degree cases;
3. performance beats V2C on Products F=128 AND F=256 under matched timing
   and does not regress arxiv materially; and
4. full memory footprint plus format preprocessing costs are reported.

Previously measured reference values (NOT measured by this branch):
Products F128 Acc 31.33ms / V2C 80.74ms; F256 Acc 70.77ms /
V2C 161.27ms. Target: <=47ms / <=106ms for new Column.

**Do not delete old code or present the outcome as an Acc-equivalent Column
until the experiment is run and verified on the target hardware.**
