#!/usr/bin/env bash

cd TAC

if [ "${time_eva_flag}" = "true" ]; then
    # TAC 时间评估
    ./time.sh 

    # 根据 csv 计算平均耗时并将结果写入文件
    # 按照 bcsc 能够掩盖 bcsr 来统计，只计算 bcsc 的时间
    python ../tools/calculate_average.py \
        --filename="${TAC_home}/result/TAC-SpMM.csv" \
        --column=3 \
        --no-header \
        --output="../baseline_time_eva.csv"
fi

if [ "${cache_eva_flag}" = "true" ]; then
    # TAC cache 命中率评估
    ./cache-hit-rate.sh

    # 根据 csv 计算平均耗时并将结果写入文件
    # l1 cache
    python ../tools/calculate_average.py \
        --filename="${TAC_home}/result/TACSpMM_cache_hit_rate.csv" \
        --column=1 \
        --no-header \
        --output="../baseline_cache_eva.csv"

    # l2 cache
    python ../tools/calculate_average.py \
            --filename="${TAC_home}/result/TACSpMM_cache_hit_rate.csv" \
        --column=2 \
        --no-header \
        --output="../baseline_cache_eva.csv"
fi
