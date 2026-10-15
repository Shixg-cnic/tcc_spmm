#!/bin/bash
cd /home/shixingguo/work/GNN/tcc_spmm/build || exit 1
/usr/local/cuda/bin/cuobjdump -sass column_partial_reduce > /tmp/p1.sass 2>&1
/usr/local/cuda/bin/cuobjdump -res-usage column_partial_reduce > /tmp/p1.res 2>&1
grep -A2 -B2 "partialPass1\|partialPass2" /tmp/p1.res | grep -E "REG|Function" | head -6
echo "--- pass1 store sites ---"
grep -c "STG.E.128\|STG.E.32" /tmp/p1.sass | head -1
awk '/partialPass1ILi16/ {f=1} f && /STG/ {print} /partialPass2/ {f=0}' /tmp/p1.sass | head -12
