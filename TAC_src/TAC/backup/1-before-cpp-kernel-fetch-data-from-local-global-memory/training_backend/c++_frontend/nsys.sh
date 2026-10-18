#!/usr/bin/env bash

#/home/gaohy/usr/local/cuda-12.4/bin/nsys profile \
#    --trace=osrt,cuda,nvtx \
#    --sample=cpu \
#    ./build/worker 0 1 1 100

/home/gaohy/usr/local/cuda-12.4/bin/nsys profile \
    --trace=osrt,cuda,nvtx \
    --sample=cpu \
    ./build/train 47 100 256 2 1 1