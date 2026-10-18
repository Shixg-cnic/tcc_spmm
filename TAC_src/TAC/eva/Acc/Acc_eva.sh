#!/usr/bin/env bash

cd Acc

if [ "${time_eva_flag}" = "true" ]; then
    # Acc 时间评估
    ./time.sh 

    # 根据 csv 计算平均耗时并将结果写入文件
    python ../tools/calculate_average.py \
        --filename="${Acc_home}/result/h100/AccSpMM-general1.csv" \
        --column=2 \
        --no-header \
        --output="../baseline_time_eva.csv"
fi

if [ "${cache_eva_flag}" = "true" ]; then
    # Acc cache 命中率评估
    ./cache-hit-rate.sh

    # 根据 csv 计算平均耗时并将结果写入文件
    # l1 cache
    python ../tools/calculate_average.py \
        --filename="${Acc_home}/result/h100/AccSpMM_cache_hit_rate.csv" \
        --column=1 \
        --no-header \
        --output="../baseline_cache_eva.csv"

    # l2 cache
    python ../tools/calculate_average.py \
        --filename="${Acc_home}/result/h100/AccSpMM_cache_hit_rate.csv" \
        --column=2 \
        --no-header \
        --output="../baseline_cache_eva.csv"
fi