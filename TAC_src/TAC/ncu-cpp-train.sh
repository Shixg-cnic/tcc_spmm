#!/usr/bin/env bash

j=570859200

for ((i=124; i<300; i++)); do
    j=$((j + 52428800))
    echo "index=$i, cache_memory=$j" | tee ncu-cpp-train.log
    
    ./run-cpp-train.sh ${j} ${i}-measure-bcsr-bcsc-cache_memory-${j}
done
