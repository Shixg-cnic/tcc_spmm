#!/usr/bin/env bash

cd $DTC_home/scripts/DTCSpMM

# 删除上次的 csv 执行结果文件
if [ -f "DTCSpMM_exe_time_and_throughput.csv" ]; then
    rm "DTCSpMM_exe_time_and_throughput.csv"
else
    echo "No DTCSpMM_exe_time_and_throughput.csv file"
fi

touch "DTCSpMM_exe_time_and_throughput.csv"

# 执行测试
source run_DTC_SpMM.sh ${epoch_number} ${iter_number}

cd ${eva_home}
