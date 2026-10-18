#!/usr/bin/env bash

sem_files=$(find /dev/shm -type f -name 'sem.fuse_sem*')
if [ -n "$sem_files" ]; then
  #echo "Found sem.sem* files, deleting them:"
  #echo "$sem_files"
  rm ${sem_files}
  echo "All sem* files have been deleted."
else
  echo "No sem* files found in the current directory."
fi

rm /dev/shm/sem.ready

rm /dev/shm/fuse_simpleIPCshm

# 结束上一次未正常结束进程
pgrep sampling_server | xargs -I{} kill {}

# products-800
python legion_server.py --dataset_path 'dataset' --dataset_name products --train_batch_size 800 --fanout [25,10] --gpu_number 1 --epoch 1 --cache_memory 38000000
#python legion_server.py --dataset_path 'dataset' --dataset_name products --train_batch_size 800 --fanout [25,10] --gpu_number 2 --epoch 1 --cache_memory 38000000
#python legion_server.py --dataset_path 'dataset' --dataset_name products --train_batch_size 800 --fanout [25,10] --gpu_number 4 --epoch 1 --cache_memory 38000000
#python legion_server.py --dataset_path 'dataset' --dataset_name products --train_batch_size 800 --fanout [25,10] --gpu_number 8 --epoch 1 --cache_memory 38000000

# products-8000(已获取，但怀疑数据有问题暂时不解决）
#python legion_server.py --dataset_path 'dataset' --dataset_name products --train_batch_size 8000 --fanout [25,10] --gpu_number 1 --epoch 1 --cache_memory 38000000

# com-friendster(800 带计算有问题；8000 已获取)
#python legion_server.py --dataset_path 'dataset' --dataset_name com-friendster --train_batch_size 800 --fanout [25,10] --gpu_number 1 --epoch 1 --cache_memory 38000000
#python legion_server.py --dataset_path 'dataset' --dataset_name com-friendster --train_batch_size 8000 --fanout [25,10] --gpu_number 1 --epoch 1 --cache_memory 38000000

# paper100m(带计算过程；800 && 8000 均已获取)
#python legion_server.py --dataset_path 'dataset' --dataset_name paper100m --train_batch_size 800 --fanout [25,10] --gpu_number 1 --epoch 1 --cache_memory 38000000
#python legion_server.py --dataset_path 'dataset' --dataset_name paper100m --train_batch_size 8000 --fanout [25,10] --gpu_number 1 --epoch 1 --cache_memory 38000000

# ukunion(非法访存，目前跳过了格式生成也获得了采样数据，但是节点和边数量都似乎不太对暂时不解决； 带计算过程； 800 && 8000 均有问题）
#python legion_server.py --dataset_path 'dataset' --dataset_name ukunion --train_batch_size 800 --fanout [25,10] --gpu_number 1 --epoch 1 --cache_memory 38000000
#python legion_server.py --dataset_path 'dataset' --dataset_name ukunion --train_batch_size 8000 --fanout [25,10] --gpu_number 1 --epoch 1 --cache_memory 38000000


#python legion_server.py --dataset_path 'dataset' --dataset_name products --train_batch_size 800 --fanout [25,10] --gpu_number 1 --epoch 1 --cache_memory ${1}

#sudo /home/gaohy/usr/local/cuda-12.4/bin/nsys profile \
#    --trace=osrt,cuda,nvtx \
#    --sample=cpu \
#    /home/gaohy/usr/local/miniconda3/envs/gaohy_legion/bin/python legion_server.py --dataset_path 'dataset' --dataset_name products --train_batch_size 800 --fanout [25,10] --gpu_number 1 --epoch 1 --cache_memory 38000000