#!/usr/bin/env bash

cd $DTC_home/scripts/TCGNN

# 删除上次的 csv 执行结果文件
if [ -f "TCGNNSpMM_cache_hit_rate.csv" ]; then
    rm "TCGNNSpMM_cache_hit_rate.csv"
else
    echo "No TCGNNSpMM_cache_hit_rate.csv file"
fi

touch "TCGNNSpMM_cache_hit_rate.csv"

# 执行测试
source ncu.sh ${epoch_number} ${iter_number}

cd ${eva_home}