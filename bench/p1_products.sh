#!/bin/bash
cd /home/shixingguo/work/GNN/tcc_spmm/build || exit 1
echo "=== products F128 FB16 ==="
./column_partial_reduce ../data/ogbn_products_full.csr 128 0 ../data/ogbn_products_full.x.bin --bench 2>&1 | grep -E "layout|correctness|pass1|pass2"
echo "=== products F128 FB16 NOSTORE ==="
./column_partial_reduce ../data/ogbn_products_full.csr 128 0 ../data/ogbn_products_full.x.bin --bench NOSTORE 2>&1 | grep -E "layout|pass1|pass2"
