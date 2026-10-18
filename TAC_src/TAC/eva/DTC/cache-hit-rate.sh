#!/usr/bin/env bash

cd $DTC_home/scripts/DTCSpMM

# 删除上次的 csv 执行结果文件
if [ -f "DTCSpMM_cache_hit_rate.csv" ]; then
    rm "DTCSpMM_cache_hit_rate.csv"
else
    echo "No DTCSpMM_cache_hit_rate.csv file"
fi

touch "DTCSpMM_cache_hit_rate.csv"

# 执行测试
source ncu.sh ${epoch_number} ${iter_number}

cd ${eva_home}