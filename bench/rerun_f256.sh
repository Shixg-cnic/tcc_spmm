#!/bin/bash
# Rerun products F256 kernel medians N times to check stability.
set -u
cd /home/shixingguo/work/GNN/tcc_spmm || exit 1
export TMPDIR=/home/shixingguo/tmp
for i in 1 2 3; do
    ./build/benchmark_cw_order data/ogbn_products_full.csr 256 0 data/ogbn_products_full.x.bin 16384 2>/dev/null \
        | grep -E 'V2C_(Original|IdentityOrder|RandomOrder|HotAnchor|BandAnchor)|Acc_Basic' \
        | cut -d, -f6,11
    echo "-- round $i done --"
done
