#!/bin/bash
# Full offline + performance pipeline for the CW-order experiment.
# 1) structure CSVs (products, arxiv); 2) benchmark 4 configs (bandRows=16384).
set -u
cd /home/shixingguo/work/GNN/tcc_spmm || exit 1
export TMPDIR=/home/shixingguo/tmp
mkdir -p results/cw_order

echo "=== analyze products ==="
./build/analyze_cw_order data/ogbn_products_full.csr > results/cw_order/products_structure.csv
echo "P1_DONE $(date +%H:%M:%S)"
echo "=== analyze arxiv ==="
./build/analyze_cw_order data/ogbn_arxiv_full.csr > results/cw_order/arxiv_structure.csv
echo "P2_DONE $(date +%H:%M:%S)"

run_bench() {
    local csr=$1 F=$2 out=$3
    local base
    base=$(basename "$csr" .csr)
    ./build/benchmark_cw_order "$csr" "$F" 0 "data/${base}.x.bin" 16384 > "$out"
}
echo "=== bench products F128 ==="
run_bench data/ogbn_products_full.csr 128 results/cw_order/products_f128.csv
echo "B1_DONE $(date +%H:%M:%S)"
echo "=== bench products F256 ==="
run_bench data/ogbn_products_full.csr 256 results/cw_order/products_f256.csv
echo "B2_DONE $(date +%H:%M:%S)"
echo "=== bench arxiv F128 ==="
run_bench data/ogbn_arxiv_full.csr 128 results/cw_order/arxiv_f128.csv
echo "B3_DONE $(date +%H:%M:%S)"
echo "=== bench arxiv F256 ==="
run_bench data/ogbn_arxiv_full.csr 256 results/cw_order/arxiv_f256.csv
echo "B4_DONE $(date +%H:%M:%S)"
echo "ALL_DONE"
