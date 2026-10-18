#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/q4_datasets
for dataset in cora pubmed dblp ogbn_arxiv ogbn_products; do
    for f in 128 256; do
        prefix="results/q4_datasets/${dataset}_f${f}"
        build/benchmark_q4_datasets "data/${dataset}_full.csr" "$f" 0 - "$prefix" > "${prefix}.stdout" 2> "${prefix}.stderr"
        gzip -f "${prefix}_cw.csv" "${prefix}_sw.csv"
    done
done
