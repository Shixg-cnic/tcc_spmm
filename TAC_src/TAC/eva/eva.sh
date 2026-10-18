#!/usr/bin/env bash

# eva.sh: 评估脚本，包含各个方法的时间和 cache 命中率评估
# source eva.sh <epoch_number> <iter_number>

# 注意事项：
# 1. 由于后续脚本需要使用 epoch_number 和 iter_number 变量，因此需要通过 source 运行
# 2. 在执行 cache hit rate 之前需要修改各个 baseline 跑 kernel 的次数，避免执行时间过程

# DTC && TCGNN npz 数据准备
export epoch_number=${1}
export iter_number=${2}

export eva_home="/work/gaohy/Legions/TAC/eva"
export TCGNN_home="/work/gaohy/Legions/TAC/baseline/DTC-SpMM_ASPLOS24"
export DTC_home="/work/gaohy/Legions/TAC/baseline/DTC-SpMM_ASPLOS24"
export FlashSparse_home="/work/gaohy/Legions/TAC/baseline/FlashSparse"
export Acc_home="/work/gaohy/Legions/TAC/baseline/Acc/spmm_compute"
export TAC_home="/work/gaohy/Legions/TAC/"

export time_eva_flag="true"
export cache_eva_flag="false"

# TAC 数据准备和评估(需要先跑完 TAC 生成数据）
TAC/TAC_eva.sh

for ((i=0; i<${epoch_number}; i++)); do
  for ((j=0; j<${iter_number}; j++)); do
    input_mtx=${i}-${j}
    output_npz=${i}-${j}

    python ./tools/DTC_mtx2npz/DTC_mtx2npz.py \
        --input_mtx="/work/gaohy/Legions/TAC/result/sample-data/${input_mtx}.mtx" \
        --output_npz="${DTC_home}/dataset/${output_npz}.npz"
  done
done

# FlashSparse npz 数据准备
for ((i=0; i<${epoch_number}; i++)); do
  for ((j=0; j<${iter_number}; j++)); do
    input_mtx=${i}-${j}
    output_npz=${i}-${j}

    python ./tools/FlashSparse_mtx2npz/FlashSparse_mtx2npz.py \
        --input_mtx="/work/gaohy/Legions/TAC/result/sample-data/${input_mtx}.mtx" \
        --output_npz="${FlashSparse_home}/dataset/${output_npz}.npz"
  done
done

# Acc 直接使用 mtx，无需进行转换

# TC-GNN 数据准备和评估
TCGNN/TCGNN_eva.sh 

# DTC 数据准备和评估
DTC/DTC_eva.sh 

# FlashSparse 数据准备和评估
FlashSparse/FlashSparse_eva.sh 

# Acc 数据准备和评估
Acc/Acc_eva.sh
