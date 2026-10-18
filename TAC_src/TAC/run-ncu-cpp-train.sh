#!/usr/bin/env bash

# 定义 server 和 consumer 的启动命令
#SERVER_CMD="./server.sh"          
#CONSUMER_CMD="/work/gaohy/Legions/TAC/training_backend/c++_frontend/run.sh" 
SERVER_CMD="./server.sh"          
CONSUMER_CMD="/work/gaohy/Legions/TAC/training_backend/c++_frontend/ncu.sh" 

# 定义日志文件
#SERVER_LOG="./profiling/server.log"
SERVER_LOG="./profiling/server.log"
CONSUMER_LOG="./profiling/train.log"

# 清理旧日志（可选）
rm -f "$SERVER_LOG" "$CONSUMER_LOG"

# 启动 server 并实时监控日志
echo "启动 server..."
stdbuf -o0 $SERVER_CMD > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

# 定义检查 server 是否就绪的函数
check_server_ready() {
    echo "等待 server 准备就绪..."
    while ! grep -q "System is ready for serving" "$SERVER_LOG"; do
        if ! ps -p $SERVER_PID > /dev/null; then
            echo "Error: Server 进程异常退出！"
            cat "$SERVER_LOG"
            exit 1
        fi
        sleep 1
    done
    echo "server 已准备就绪"
}

# 等待 server 准备就绪
check_server_ready

# 启动 consumer
echo "启动 trainer..."
stdbuf -o0 $CONSUMER_CMD > "$CONSUMER_LOG" 2>&1 &
CONSUMER_PID=$!

# 等待所有进程完成（可选）
wait $SERVER_PID $CONSUMER_PID

# 检查进程退出状态（可选）
if [ $? -ne 0 ]; then
    echo "Error: 进程异常退出"
    exit 1
fi

echo "所有任务完成！"