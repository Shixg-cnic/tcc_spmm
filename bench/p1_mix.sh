#!/bin/bash
cd /home/shixingguo/work/GNN/tcc_spmm/build || exit 1
/usr/local/cuda/bin/cuobjdump -sass column_partial_reduce > /tmp/p1.sass 2>&1
awk '/partialPass1ILi16/ {f=1} /partialPass2/ {f=0} f' /tmp/p1.sass > /tmp/pass1.body
echo "pass1 instruction mix:"
for op in STG.E.128 STG.E LDG.E LDS BAR.SYNC HMMA.SS SHFL.IDX IMAD.WIDE LDG.E.128 IADD3; do
  n=$(grep -o "$op" /tmp/pass1.body | wc -l); echo "$op: $n";
done
echo "pass1 total lines: $(wc -l < /tmp/pass1.body)"
