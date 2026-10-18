#!/usr/bin/env bash

cd $TCGNN_home/scripts/TCGNN

# 删除上次的 csv 执行结果文件
if [ -f "TCGNNSpMM_exe_time_and_throughput.csv" ]; then
    rm "TCGNNSpMM_exe_time_and_throughput.csv"
else
    echo "No TCGNNSpMM_exe_time_and_throughput.csv"
fi

touch "TCGNNSpMM_exe_time_and_throughput.csv"

# 执行测试
source run_TCGNN_SpMM.sh ${epoch_number} ${iter_number}

cd ${eva_home}
