#!/bin/bash
F=/home/shixingguo/work/GNN/tcc_spmm/third_party/Acc-SpMM/include/mma_tf32.cuh
echo "X load sites: $(grep -c 'load_fp32_from_global(d_MatB' $F)"
grep -n 'load_fp32_from_global(d_MatB' $F | sed 's/:.*load/: load/' | head -14
