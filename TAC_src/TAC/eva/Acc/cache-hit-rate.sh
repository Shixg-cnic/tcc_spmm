#!/usr/bin/env bash

cd $Acc_home/result/h100

# 删除上次的 csv 执行结果文件
if [ -f "AccSpMM_cache_hit_rate.csv" ]; then
    rm "AccSpMM_cache_hit_rate.csv"
else
    echo "No AccSpMM_cache_hit_rate.csv file"
fi

touch "AccSpMM_cache_hit_rate.csv"

# 执行测试
cd $Acc_home

ncu_profiler() {
    dataset=$1

    ncu \
    --replay-mode kernel \
    --set full \
    --target-processes all \
    --kernel-name regex:"tf32_computeX128TransposePipe2.*" \
    --call-stack \
    --nvtx \
    --import-source yes \
    -o ncu_profiler \
    ./mma_tf32 "${eva_home}/../result/sample-data/${dataset}.mtx" 

    ncu -i ncu_profiler.ncu-rep --csv --page raw &> ncu_profiler.csv

    rm ncu_profiler.ncu-rep

    /work/gaohy/Legions/TAC/eva/tools/csv_tools/extract_data_from_csv ncu_profiler.csv \
      /work/gaohy/Legions/TAC/baseline/Acc/spmm_compute/result/h100/AccSpMM_cache_hit_rate.csv\
      "Kernel Name" \
      l1tex__t_sector_hit_rate.pct \
      lts__t_sector_hit_rate.pct

    rm ncu_profiler.csv
}

for ((i=0; i<${epoch_number}; i++)); do
  for ((j=0; j<${iter_number}; j++)); do
    dataset=${i}-${j}
    ncu_profiler ${dataset}
  done
done

cd ${eva_home}