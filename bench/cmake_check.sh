#!/bin/bash
grep -n "acc_reference\|convertAccRow\|launchAccRow" /home/shixingguo/work/GNN/tcc_spmm/bench/benchmark_int4.cu | head
echo "---- cmake ----"
grep -nE "column_fast|benchmark_int4|acc_reference|add_executable|add_library" /home/shixingguo/work/GNN/tcc_spmm/CMakeLists.txt
