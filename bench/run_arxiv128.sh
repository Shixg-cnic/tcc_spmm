#!/bin/bash
cd /home/shixingguo/work/GNN/tcc_spmm/build || exit 1
echo "=== arxiv F128 ==="
./benchmark_fast_int4 ../data/ogbn_arxiv_full.csr 128 0 ../data/ogbn_arxiv_full.x.bin 2>&1 | tail -8
