#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/q4
for dataset in arxiv products; do
    for features in 128 256; do
        ./build/benchmark_q4 "data/ogbn_${dataset}_full.csr" "$features" "${1:-0}" \
            "data/ogbn_${dataset}_full.x.bin" \
            > "results/q4/${dataset}_f${features}.csv" \
            2> "results/q4/${dataset}_f${features}.stderr"
    done
done
python3 bench/summarize_q4.py
