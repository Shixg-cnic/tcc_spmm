#!/bin/bash
# Show the address-computation dependency chain before the C0 REDG sites.
set -u
cd /home/shixingguo/work/GNN/tcc_spmm/build || exit 1
CU=/usr/local/cuda/bin
$CU/cuobjdump -sass column_output_diagnosis > /tmp/all_sass.txt 2>&1
awk '/Function : .*columnKernelILi0ELb0E/ {f=1}
     f && /\/\*0/ {line[NR]=$0}
     f && /REDG/ && !seen {for(i=NR-14;i<=NR;i++) print line[i]; seen=1; exit}' /tmp/all_sass.txt
