#!/usr/bin/env bash

#sem_files=$(find /dev/shm -type f -name 'sem.fuse_sem*')
#if [ -n "$sem_files" ]; then
#  #echo "Found sem.sem* files, deleting them:"
#  #echo "$sem_files"
#  rm ${sem_files}
#  echo "All sem* files have been deleted."
#else
#  echo "No sem* files found in the current directory."
#fi
#
#rm /dev/shm/sem.ready

# 结束上一次未正常结束进程
pgrep -f /home/user/anaconda3/envs/gaohy_legion/bin/python | xargs -I{} kill {}

# ukunion
#python training_backend/legion_graphsage.py --class_num 2  --features_num 128 --hidden_dim 256 --hops_num 2 --gpu_number 8 --epoch 1

# products
/home/gaohy/usr/local/miniconda3/envs/gaohy_legion/bin/python training_backend/backward_model.py --class_num 47 --features_num 100 --hidden_dim 256 --hops_num 2 --gpu_number 1 --epoch 1
#python training_backend/backward_model.py --class_num 47  --features_num 100 --hidden_dim 256 --hops_num 2 --gpu_number 2 --epoch 1
#python training_backend/backward_model.py --class_num 47  --features_num 100 --hidden_dim 256 --hops_num 2 --gpu_number 4 --epoch 1
#python training_backend/backward_model.py --class_num 47 --features_num 100 --hidden_dim 256 --hops_num 2 --gpu_number 8 --epoch 1

#CUDA_LAUNCH_BLOCKING=1 cuda-gdb --silent --args python training_backend/backward_model.py --class_num 47  --features_num 100 --hidden_dim 256 --hops_num 2 --gpu_number 1 --epoch 1
