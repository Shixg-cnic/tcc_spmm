#!/usr/bin/env bash

# 删除上次的 csv 执行结果文件
cd $FlashSparse_home/result/FlashSparse/spmm

# 删除上次的 csv 执行结果文件
if [ -f "FlashSparseSpMM_cache_hit_rate.csv" ]; then
    rm "FlashSparseSpMM_cache_hit_rate.csv"
else
    echo "No FlashSparseSpMM_cache_hit_rate.csv file"
fi

touch "FlashSparseSpMM_cache_hit_rate.csv"

# 执行测试
cd $FlashSparse_home/eva/kernel/spmm
source ncu.sh ${epoch_number} ${iter_number}

cd ${eva_home}
