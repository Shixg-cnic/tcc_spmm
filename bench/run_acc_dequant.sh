#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/acc_dequant build/tmp
export TMPDIR="$PWD/build/tmp"
cmake -S . -B build > results/acc_dequant/configure.log 2>&1
cmake --build build --target benchmark_acc_dequant -j 4 > results/acc_dequant/build.log 2>&1
nvidia-smi > results/acc_dequant/environment.txt
/home/shixingguo/miniforge3/envs/gnn/bin/nvcc --version >> results/acc_dequant/environment.txt
sha256sum third_party/Acc-SpMM/include/mma_tf32.cuh third_party/Acc-SpMM/include/ptx_tf32.hpp data/ogbn_arxiv_full.csr data/ogbn_products_full.csr data/native_features/ogbn-arxiv.x.bin data/native_features/ogbn-products.x.bin > results/acc_dequant/input_sha256.txt
for name in arxiv products; do
    for f in 128 256 512; do
        echo "Running $name F=$f"
        build/benchmark_acc_dequant "data/ogbn_${name}_full.csr" "data/native_features/ogbn-${name}.x.bin" "$f" "${DEVICE:-0}" "results/acc_dequant/${name}_f${f}" > "results/acc_dequant/${name}_f${f}.stdout.log" 2> "results/acc_dequant/${name}_f${f}.stderr.log"
    done
done
python3 bench/summarize_acc_dequant.py
