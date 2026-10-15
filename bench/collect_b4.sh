#!/bin/bash
# collect the batch-4 CSV outputs into results/kernel_rebuild
cd /home/shixingguo/work/GNN/tcc_spmm/build || exit 1
mkdir -p ../results/kernel_rebuild
for cfg in "arxiv 128" "arxiv 256" "products 128" "products 256"; do
  set -- $cfg
  out=../results/kernel_rebuild/benchmark_fast_int4_${1}_f${2}.csv
  ./benchmark_fast_int4 ../data/ogbn_${1}_full.csr $2 0 ../data/ogbn_${1}_full.x.bin > "$out" 2>&1
  echo "wrote $out ($(grep -c ",$1" "$out") rows)"
done
