#!/bin/bash
# CW-order benchmark: 4 configs x bandRows=16384, per-config CSV. Run on GPU 0
# exclusively (check nvidia-smi first).
set -u
cd /home/shixingguo/work/GNN/tcc_spmm || exit 1
export TMPDIR=/home/shixingguo/tmp
mkdir -p results/cw_order
run() {
    local csr=$1 F=$2 out=$3
    local base
    base=$(basename "$csr" .csr)          # ogbn_products_full
    ./build/benchmark_cw_order "$csr" "$F" 0 "data/${base}.x.bin" 16384 > "$out"
}
run data/ogbn_products_full.csr 128 results/cw_order/products_f128.csv
run data/ogbn_products_full.csr 256 results/cw_order/products_f256.csv
run data/ogbn_arxiv_full.csr 128 results/cw_order/arxiv_f128.csv
run data/ogbn_arxiv_full.csr 256 results/cw_order/arxiv_f256.csv
echo "ALL_DONE"
