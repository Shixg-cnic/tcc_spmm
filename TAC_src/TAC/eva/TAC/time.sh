#!/usr/bin/env bash

# 删除上次的 csv 执行结果文件
cd $TAC_home/result
if [ -f "TAC-SpMM.csv" ]; then
    rm "TAC-SpMM.csv"
else
    echo "No TAC-SpMM.csv file"
fi

touch "TAC-SpMM.csv"

# 执行测试
cd $TAC_home
./run-cpp-train.sh

cd ${eva_home}
