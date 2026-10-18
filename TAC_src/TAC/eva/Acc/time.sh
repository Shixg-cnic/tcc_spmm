#!/usr/bin/env bash

# 删除上次的 csv 执行结果文件
cd $Acc_home/result/h100

if [ -f "AccSpMM-general1.csv" ]; then
    rm "AccSpMM-general1.csv"
else
    echo "No AccSpMM-general1.csv file"
fi

touch "AccSpMM-general1.csv"

# 执行测试
cd $Acc_home

for ((i=0; i<${epoch_number}; i++)); do
  for ((j=0; j<${iter_number}; j++)); do
    dataset=${i}-${j}
    ./mma_tf32 "${eva_home}/../result/sample-data/${dataset}.mtx" 
  done
done

cd ${eva_home}
