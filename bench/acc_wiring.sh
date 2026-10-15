#!/bin/bash
grep -n 'launchAccRowSpMM\|convertAccRow\|RowBitTCF\|acc\.' /home/shixingguo/work/GNN/tcc_spmm/bench/benchmark_fp32.cu | head -20
echo ---
head -40 /home/shixingguo/work/GNN/tcc_spmm/tests/row_format.cuh
