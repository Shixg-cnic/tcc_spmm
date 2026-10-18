#!/usr/bin/env bash

# 单独执行 server.sh，然后执行此 profile.sh

# --import-source yes：将源代码打包进 .ncu-rep 文件，便于 ncu gui 程序能够定位到源代码

#/home/gaohy/usr/local/cuda-12.4/bin/ncu --replay-mode kernel --set full --metrics all --kernel-name tf32_computeX128TransposePipe2 --target-processes all --call-stack --nvtx --import-source yes -o ${1} /home/gaohy/Legions/TAC/training_backend/c++_frontend/build/worker 0 1 1 100
#/home/gaohy/usr/local/cuda-12.4/bin/ncu --replay-mode kernel --set full --kernel-name tf32_computeX128TransposePipe2 --target-processes all --call-stack --nvtx --import-source yes -o ${1} /home/gaohy/Legions/TAC/training_backend/c++_frontend/build/worker 0 1 1 100

#sudo /home/gaohy/Legions/TAC/training_backend/c++_frontend/build/worker 0 1 1 100

/home/gaohy/usr/local/cuda-12.4/bin/ncu --replay-mode kernel --set full --target-processes all --call-stack --nvtx --import-source yes -o ${1} /home/gaohy/Legions/TAC/training_backend/c++_frontend/build/worker 0 1 1 100
