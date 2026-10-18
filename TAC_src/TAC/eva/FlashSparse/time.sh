#!/usr/bin/env bash

# 删除上次的 csv 执行结果文件
cd $FlashSparse_home/result/FlashSparse/spmm

if [ -f "spmm_tf32_128.csv" ]; then
    rm "spmm_tf32_128.csv"
else
    echo "No spmm_tf32_128.csv file"
fi

touch "spmm_tf32_128.csv"

# 执行测试
cd $FlashSparse_home/eva/kernel/spmm
source test_spmm_shell.sh ${epoch_number} ${iter_number}

cd ${eva_home}
