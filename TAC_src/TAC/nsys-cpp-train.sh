# 由 c++ 仅完成算子调用，避免 model 的影响

nsys profile \
    --trace=osrt,cuda,nvtx \
    --sample=cpu \
    ./run-cpp-train.sh

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

