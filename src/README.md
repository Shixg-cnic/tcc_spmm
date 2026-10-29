# 源码

- `format/matrix_market.cpp`：读取 `.mtx` 为 CSR。
- `format/csr.cpp`：行内排序、合并重复坐标。
- `format/sw32.cpp`：生成 SW32；保留 RW 分桶与 panel 顺序，将重 SW 划分为最多 384 panel 的任务，并建立紧凑部分和归并索引。
- `format/int8_quantize.cu`、`format/pack_int8.cu`：量化与 packed16 打包。
- `kernel/spmm.cu`：F128/F256 启动入口，接收显式 workspace。
- `kernel/spmm_kernel.cuh`：有界 panel 任务；独占部分对 A/source ID 使用异步双缓冲，跨 RW 部分复用 X；C 在寄存器中累加，重 SW 由最后完成的 CTA 在同一个 kernel 内归并。
- `main.cu`：加载数据，上传任务描述，分配并初始化 workspace，计时运行。

运行方法、workspace 生命周期与实测结果见根目录 README。
