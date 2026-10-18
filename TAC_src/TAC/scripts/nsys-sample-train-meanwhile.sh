# 采用新的采样-训练交互逻辑，利用采样进程预采样过程的时间开销掩盖训练进程的一些预备操作时间开销

/home/gaohy/usr/local/cuda-12.4/bin/nsys profile \
    --trace=osrt,cuda,nvtx \
    --sample=cpu \
    ./run-sample-train-meanwhile.sh

#nsys profile \
#  --stats=false \
#  --trace=cuda,nvtx,osrt \
#  --sample=system-wide \
#  --cudabacktrace=all:500 \
#  --cuda-memory-usage=true \
#  --cuda-um-cpu-page-faults=true \
#  --cuda-um-gpu-page-faults=true \
#  --cuda-graph-trace=node \
#  --gpuctxsw=true \
#  --cpuctxsw=system-wide \
#  --gpu-metrics-device=all \
#  --osrt-threshold=100 \
#  --osrt-backtrace-threshold=5000 \
#  --osrt-backtrace-stack-size=8192 \
#  --event-sample=system-wide \
#  --python-sampling=true \
#  --python-sampling-frequency=1000 \
#  ./run.sh

