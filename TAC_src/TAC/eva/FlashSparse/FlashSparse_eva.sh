#!/usr/bin/env bash

compile() {
    cd ${FlashSparse_home}/FlashSparse
    bash compile.sh
}

cd FlashSparse

if [ "${time_eva_flag}" = "true" ]; then

    # FlashSparse 时间评估
    ./time.sh

    # 根据 csv 计算平均耗时并将结果写入文件
    python ../tools/calculate_average.py \
        --filename="${FlashSparse_home}/result/FlashSparse/spmm/spmm_tf32_128.csv" \
        --column=1 \
        --no-header \
        --output="../baseline_time_eva.csv"

    # =============================== 命中率 ===============================
    # 修改 kernel 执行次数并重新编译代码
    #/work/gaohy/Legions/TAC/baseline/FlashSparse/FlashSparse/SpMM/src/spmmKernel.cu
    #define WarmUp_EXE_TIME 10 -> 0

    #/work/gaohy/Legions/TAC/baseline/FlashSparse/eva/kernel/spmm/spmm_tf32_test_args.py
    #epoches = 10 -> 1

    #sed -i '/^#define[[:space:]]\+EXE_TIME[[:space:]]\+[0-9]\+/s/[0-9]\+/1000/' filename.h

    #compile
fi

if [ "${cache_eva_flag}" = "true" ]; then
    # FlashSparse cache 命中率评估
    ./cache-hit-rate.sh

    # 根据 csv 计算平均耗时并将结果写入文件
    # l1 cache
    python ../tools/calculate_average.py \
        --filename="${FlashSparse_home}/result/FlashSparse/spmm/FlashSparseSpMM_cache_hit_rate.csv" \
        --column=1 \
        --no-header \
        --output="../baseline_cache_eva.csv"

    # l2 cache
    python ../tools/calculate_average.py \
        --filename="${FlashSparse_home}/result/FlashSparse/spmm/FlashSparseSpMM_cache_hit_rate.csv" \
        --column=2 \
        --no-header \
        --output="../baseline_cache_eva.csv"

    # 执行次数复原
fi