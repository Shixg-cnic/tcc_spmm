#!/bin/bash
# BandAnchor bandRows sweep on products F128 (16384 already measured).
set -u
cd /home/shixingguo/work/GNN/tcc_spmm || exit 1
export TMPDIR=/home/shixingguo/tmp
for band in 8192 32768; do
    ./build/benchmark_cw_order data/ogbn_products_full.csr 128 0 data/ogbn_products_full.x.bin "$band" 2>/dev/null \
        | grep -E 'V2C_(Original|IdentityOrder|RandomOrder|HotAnchor|BandAnchor)|Acc_Basic' \
        | cut -d, -f3,6,7,11,15
    echo "-- band $band done --"
done
