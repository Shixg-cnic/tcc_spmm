#!/usr/bin/env bash

# 单独执行 server.sh，然后执行此 profile.sh

# --import-source yes：将源代码打包进 .ncu-rep 文件，便于 ncu gui 程序能够定位到源代码

#/home/gaohy/usr/local/cuda-12.4/bin/ncu --replay-mode kernel --set full --metrics all --kernel-name tf32_computeX128TransposePipe2 --target-processes all --call-stack --nvtx --import-source yes -o ${1} /home/gaohy/Legions/TAC/training_backend/c++_frontend/build/worker 0 1 1 100
#/home/gaohy/usr/local/cuda-12.4/bin/ncu --replay-mode kernel --set full --kernel-name tf32_computeX128TransposePipe2 --target-processes all --call-stack --nvtx --import-source yes -o ${1} /home/gaohy/Legions/TAC/training_backend/c++_frontend/build/worker 0 1 1 100

#sudo /home/gaohy/Legions/TAC/training_backend/c++_frontend/build/worker 0 1 1 100

base_dir="/work/gaohy/Legions/TAC/training_backend/c++_frontend/"

# first layer BCSR and BCSC
#ncu \
#    --replay-mode kernel \
#    --set full \
#    --target-processes all \
#    --kernel-name regex:"tf32_computeX128TransposePipe2_BCS(R|C)_forward_first_layer" \
#    --call-stack \
#    --nvtx \
#    --import-source yes \
#    -o ${1} \
#    ${base_dir}/build/worker_with_model 0 47 100 256 2 1 1

# first layer BCSR
ncu \
    --replay-mode kernel \
    --set full \
    --target-processes all \
    --kernel-name regex:"tf32_computeX128TransposePipe2_BCSR_forward_first_layer" \
    --call-stack \
    --nvtx \
    --import-source yes \
    -o ncu_profiler \
    ${base_dir}/build/worker_with_model 0 47 100 256 2 1 1


# first layer BCSC
#ncu \
#    --replay-mode kernel \
#    --set full \
#    --target-processes all \
#    --kernel-name regex:"tf32_computeX128TransposePipe2_BCSC_forward_first_layer" \
#    --call-stack \
#    --nvtx \
#    --import-source yes \
#    -o ${1} \
#    build/worker_with_model 0 47 100 256 2 1 1

# first layer BCSR acc
#ncu \
#    --replay-mode kernel \
#    --set full \
#    --target-processes all \
#    --kernel-name regex:"tf32_computeX128TransposePipe2_BCSR_forward_first_layer_acc" \
#    --call-stack \
#    --nvtx \
#    --import-source yes \
#    -o ${1} \
#    build/worker_with_model 0 47 100 256 2 1 1

# first layer BCSC 混合采集
#ncu \
#    --replay-mode kernel \
#    --set full \
#    --target-processes all \
#    --kernel-name regex:"tf32_computeX128TransposePipe2_BCSR_forward_first_layer.*" \
#    --call-stack \
#    --nvtx \
#    --import-source yes \
#    -o ${1} \
#    build/worker_with_model 0 47 100 256 2 1 1

# BCSC
#ncu \
#    --replay-mode kernel \
#    --set full \
#    --target-processes all \
#    --kernel-name regex:"tf32_computeX128TransposePipe2_BCSC_forward_first_layer" \
#    --call-stack \
#    --nvtx \
#    --import-source yes \
#    -o ${1} \
#    build/worker_with_model 0 47 100 256 2 1 1

# ncu-rep 转 csv
#ncu -i ${1}.ncu-rep --csv --page raw  --metrics l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct &> ${1}.csv
#ncu -i ${base_dir}/profiling/${1}.ncu-rep --csv --page raw &> ${base_dir}/profiling/${1}.csv

ncu -i ncu_profiler.ncu-rep --csv --page raw &> ncu_profiler.csv

rm ncu_profiler.ncu-rep

/work/gaohy/Legions/TAC/eva/tools/csv_tools/extract_data_from_csv ncu_profiler.csv \
  /work/gaohy/Legions/TAC/result/TACSpMM_cache_hit_rate.csv\
  "Kernel Name" \
  l1tex__t_sector_hit_rate.pct \
  lts__t_sector_hit_rate.pct

rm ncu_profiler.csv

# 计算平均值
#./compute_multi_category_metrics_average profiling/${1}.csv \
#    l1tex__t_sector_hit_rate.pct \
#    lts__t_sector_hit_rate.pct 

#./compute_multi_category_metrics_average profiling/${1}.csv \
#    "Kernel Name" \
#    dram__bytes_read.sum \
#    dram__bytes_write.sum \
#    &> ${base_dir}/profiling/${1}.txt


# 计算混合 L1 命中率
#l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
#l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum
#
#l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum
#l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_hit.sum
#
#(l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum + l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_hit.sum) / 
#(l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum + l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum)

#./compute_hybrid_l1_cache_hit_rate profiling/${1}.csv \
#    "Kernel Name"  \
#    l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum \
#    l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_hit.sum \
#    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum \
#    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum
#

# 计算混合 L2 命中率
#lts__t_sectors_srcunit_tex_lookup_hit.sum
#lts__t_sectors_srcunit_tex_lookup_miss.sum
#
#lts__t_sectors_srcunit_ltcfabric_lookup_hit.sum
#lts__t_sectors_srcunit_ltcfabric_lookup_miss.sum
#
#(lts__t_sectors_srcunit_tex_lookup_hit.sum + lts__t_sectors_srcunit_ltcfabric_lookup_hit.sum) / 
#(lts__t_sectors_srcunit_tex_lookup_hit.sum + lts__t_sectors_srcunit_tex_lookup_miss.sum + lts__t_sectors_srcunit_ltcfabric_lookup_hit.sum + lts__t_sectors_srcunit_ltcfabric_lookup_miss.sum)

#${base_dir}/compute_hybrid_l2_cache_hit_rate ${base_dir}/profiling/${1}.csv \
#    "Kernel Name"   \
#    lts__t_sectors_srcunit_tex_lookup_hit.sum \
#    lts__t_sectors_srcunit_ltcfabric_lookup_hit.sum \
#    lts__t_sectors_srcunit_tex_lookup_miss.sum \
#    lts__t_sectors_srcunit_ltcfabric_lookup_miss.sum \
#    &> ${base_dir}/profiling/${1}.txt