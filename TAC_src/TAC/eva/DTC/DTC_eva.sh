#!/usr/bin/env bash

cd DTC

if [ "${time_eva_flag}" = "true" ]; then
    # DTC 时间评估
    ./time.sh 

    # 根据 csv 计算平均耗时并将结果写入文件
    python ../tools/calculate_average.py \
        --filename="${DTC_home}/scripts/DTCSpMM/DTCSpMM_exe_time_and_throughput.csv" \
        --column=3 \
        --no-header \
        --output="../baseline_time_eva.csv"
fi

if [ "${cache_eva_flag}" = "true" ]; then
    # DTC cache 命中率评估
    ./cache-hit-rate.sh

    # 根据 csv 计算平均耗时并将结果写入文件
    # l1 cache
    python ../tools/calculate_average.py \
        --filename="${DTC_home}/scripts/DTCSpMM/DTCSpMM_cache_hit_rate.csv" \
        --column=1 \
        --no-header \
        --output="../baseline_cache_eva.csv"

    # l2 cache
    python ../tools/calculate_average.py \
        --filename="${DTC_home}/scripts/DTCSpMM/DTCSpMM_cache_hit_rate.csv" \
        --column=2 \
        --no-header \
        --output="../baseline_cache_eva.csv"
fi