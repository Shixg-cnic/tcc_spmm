#!/bin/bash
cd /home/shixingguo/work/GNN/tcc_spmm/build || exit 1
for cfg in "arxiv 256" "products 128" "products 256"; do
  set -- $cfg
  echo "=== $1 F$2 ==="
  ./benchmark_fast_int4 ../data/ogbn_${1}_full.csr $2 0 ../data/ogbn_${1}_full.x.bin 2>&1 | tail -7
done
