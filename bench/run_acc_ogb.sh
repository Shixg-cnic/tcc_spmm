#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export TMPDIR="$PWD/build/tmp"
DEVICE="${DEVICE:-1}"
OUT="$PWD/results/acc_dequant_ogb"
mkdir -p "$OUT"
# Build first with: TMPDIR=$PWD/build/tmp cmake --build build --target benchmark_acc_dequant -j4
nvidia-smi > "$OUT/environment.txt"
/home/shixingguo/miniforge3/envs/gnn/bin/nvcc --version >> "$OUT/environment.txt"
for name in arxiv products proteins; do
    matrix="data/ogbn_${name}_full.csr"
    features="data/native_features/ogbn-${name}.x.bin"
    if [[ "$name" == proteins ]]; then
        matrix=data/ogb_extended/ogbn-proteins.csr
        features=data/ogb_extended/ogbn-proteins.x.bin
    fi
    for f in 64 128 256 512; do
        prefix="$OUT/${name}_f${f}"
        echo "Running $name F=$f GPU=$DEVICE"
        nvidia-smi > "${prefix}_gpu_before.txt"
        build/benchmark_acc_dequant "$matrix" "$features" "$f" "$DEVICE" "$prefix" > "${prefix}.stdout.log" 2> "${prefix}.stderr.log"
        nvidia-smi > "${prefix}_gpu_after.txt"
    done
done
python3 bench/summarize_acc_ogb.py
