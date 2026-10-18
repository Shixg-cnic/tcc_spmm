#!/usr/bin/env bash

cd $TAC_home/result

# 删除上次的 csv 执行结果文件
if [ -f "TACSpMM_cache_hit_rate.csv" ]; then
    rm "TACSpMM_cache_hit_rate.csv"
else
    echo "No TACSpMM_cache_hit_rate.csv file"
fi

touch "TACSpMM_cache_hit_rate.csv"

# 执行测试
cd $TAC_home

./run-ncu-cpp-train.sh 

cd ${eva_home}