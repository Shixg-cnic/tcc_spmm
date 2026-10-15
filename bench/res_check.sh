#!/bin/bash
cd /home/shixingguo/work/GNN/tcc_spmm/build || exit 1
/usr/local/cuda/bin/cuobjdump -res-usage column_fast_test > /tmp/res_fast.txt 2>&1
awk '/fastKernel/ {f=$0} /REG/ && f {print f " -> " $0; f=""}' /tmp/res_fast.txt
