# 源码阅读顺序

1. `main.cu`：从 `run()` 看完整流程：读矩阵、生成格式、上传、准备 INT8 特征、计时。`prepareFeatures()` 不在计时区间内，`benchmark()` 保留 10 次预热与 10 次计时。
2. `device_matrix.cuh`：集中管理矩阵数组与拆分任务的 workspace，向 kernel 提供 `view()` 和 `workspace()`。
3. `format/sw32.cpp`：`convertCsrToSw32()` 生成 SW32；`buildPanelTasks()` 将重 SW 划分为最多 384 panel 的任务，并建立部分和归并索引。
4. `kernel/spmm.cu`：只选择 F128/F256 并启动 kernel；每 CTA 128 线程。
5. `kernel/spmm_kernel.cuh`：按四步阅读 `spmmInt8Kernel()`：独占 RW 的 A/ID 异步双缓冲、跨 RW 复用 X、写回任务结果、最后一个 CTA 归并。C 始终在寄存器中累加。

其他文件：

- `format/matrix_market.cpp`、`format/csr.cpp`：读取 `.mtx`、行内排序及合并重复坐标。
- `format/int8_quantize.cu`、`format/pack_int8.cu`：按源节点量化与 packed16 打包。
- `../include/cuda_support.cuh`：统一 CUDA 错误报告和设备内存生命周期。
- `../include/format/sw32.h`、`../include/spmm.cuh`：数据结构与库入口。

CUDA 入口直接调用，不再逐层传递 `cudaError_t`。启动失败抛出异常，设备执行保持异步；读取结果前仍需同步。输入有效性在文件/格式边界检查，设备数组和 workspace 由调用方按接口约定提供。占用率与资源查询不是计算路径，已移除。

运行命令、workspace 生命周期与历史实测结果见根目录 README。
