#!/bin/bash
cd /home/shixingguo/work/GNN/tcc_spmm/build || exit 1
for m in SCALAR NOSTORE plain; do
  echo "=== $m ==="
  if [ "$m" = "plain" ]; then
    ./column_partial_reduce ../data/ogbn_arxiv_full.csr 128 0 ../data/ogbn_arxiv_full.x.bin --bench 2>&1 | grep -E "correctness|pass1|pass2"
  else
    ./column_partial_reduce ../data/ogbn_arxiv_full.csr 128 0 ../data/ogbn_arxiv_full.x.bin --bench "$m" 2>&1 | grep -E "correctness|pass1|pass2"
  fi
done
