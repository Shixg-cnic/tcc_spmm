#!/usr/bin/env bash

#latest_folder=$(ls /work/gaohy/Legion_fuse/logs | sort -r | head -n 1)

#log_dir="/work/gaohy/Legion_fuse/logs/${latest_folder}"

#CUDA_LAUNCH_BLOCKING=1 cuda-gdb --silent --args build/train 47 100 256 2 1 10 | tee "${log_dir}/train.log"

#CUDA_LAUNCH_BLOCKING=1 build/train 47 100 256 2 1 10 | tee "${log_dir}/train.log" 
#build/train 47 100 256 2 4 1 | tee "/work/gaohy/Legions/TAC/profiling/train.log" 


#sem_files=$(find /dev/shm -type f -name 'sem.fuse_sem*')
#if [ -n "$sem_files" ]; then
#    #echo "Found sem.sem* files, deleting them:"
#    #echo "$sem_files"
#    sudo rm ${sem_files}
#    echo "All sem* files have been deleted."
#else
#    echo "No sem* files found in the current directory."
#fi
#
#sudo rm /dev/shm/sem.ready

/home/gaohy/Legions/TAC/training_backend/c++_frontend/build/train 47 100 256 2 1 1
#/home/gaohy/Legions/TAC/training_backend/c++_frontend/build/train 47 100 256 2 2 1
#/home/gaohy/Legions/TAC/training_backend/c++_frontend/build/train 47 100 256 2 4 1
#/home/gaohy/Legions/TAC/training_backend/c++_frontend/build/train 47 100 256 2 8 1

#sudo /home/gaohy/Legions/TAC/training_backend/c++_frontend/build/worker 0 1 1 100
#/home/gaohy/Legions/TAC/training_backend/c++_frontend/build/worker 0 1 1 100
