# TAC BCSC 独立复现

独立可执行文件 `tac_bcsc_repro`，不链接项目的 SpMM kernel、PyTorch、IPC 或采样服务。复用项目的 CSR 文件读取和测试工具，单独实现 CPU 格式转换与 GPU CSR 参考计算。

## 复现对象

`TAC_src/TAC/training_backend/c++_frontend/3rd/spmm_cnic_gat_edit_denseC/src/spmm_kernel.cu` 中的 `tf32_computeX128TransposePipe2_BCSC_forward_first_layer`。`original_kernel.cuh` 保留该函数的完整原文。

- 一个 CTA 处理一个固定的 8 源列窗口；每个 warp 处理 32 个特征。
- 按窗口内目标行排序、去重，每 8 个目标行形成一个 tile。
- bitmap 位号为 `local_output_row * 8 + source_column % 8`。
- 输入特征加载到寄存器后在整个列窗口内复用。
- 共享内存双缓冲解码 bitmap、预取输出行号；TF32 `m16n8k8`；每个 tile 原子累加输出。
- 只支持无权图（非零值必须为 1），不包含 Q4 分组或生命周期优化。

## 适配及边界

1. 原 `ptx_tf32.h` 把 FP32 转换输入声明为整数寄存器，并改写 input-only 操作数，CUDA 12.8 编译失败。`ptx_compat.cuh` 使用 FP32 输入、独立 TF32 输出寄存器，保持 RNA 舍入及 MMA 指令；内存指令补充编译器 memory clobber。原失败日志保存在 `results/tac_bcsc/original_compile_failure.log`。
2. 原 kernel 无条件预取第一个 tile。独立转换器给空窗口补一个全零 bitmap tile，行号为 `UINT32_MAX`，防止越界；原始导入源码不修改。
3. 启动维度沿用原 dispatcher 的 `(32, F/32)`。限制 F 为 `[64,512]` 内的 32 倍数；F=32 的一个 warp 不足以初始化 64 个共享内存元素。原 dispatcher 对 F>512 也没有实现。
4. 输出每次必须清零。`clear_kernel_ms` 包含清零；`kernel_ms` 只测原子累加 kernel，重复调用会累积输出，仅作性能诊断。
5. 输入矩阵必须是非零维度，转换器拒绝非单位权重。大图加载遵循现有 CSR canonicalization 规则；合并后非单位权重也拒绝。

## 构建与运行

在仓库根目录运行（CUDA 架构按实际 GPU 设置）：

```bash
mkdir -p build/tmp
TMPDIR="$PWD/build/tmp" cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=120
TMPDIR="$PWD/build/tmp" cmake --build build --target tac_bcsc_repro -j4
ctest --test-dir build -R '^tac_bcsc_correctness$' --output-on-failure
build/tac_bcsc_repro data/ogbn_arxiv_full.csr 128 data/ogbn_arxiv_full.x.bin
build/tac_bcsc_repro data/ogbn_products_full.csr 256 data/ogbn_products_full.x.bin
```

不提供 X 文件时使用固定种子的随机特征。固定使用 GPU 0。

25 组小图测试覆盖全空图、空窗口、单元素、非整齐行列尾部、多 tile、跨窗口输出冲突，以及 F=64/96/128/256/512。同时验证 CPU CSR 和 GPU CSR 参考结果，检查重复清零后调用，并检查拒绝带权图。全图使用独立逐行 GPU CSR（TF32 RNA 输入、double 累加）检查全部输出。

计时使用 CUDA Graph：10 次预热，每个 graph 10 次调用，5 次测量取中位数。格式转换耗时为 CPU 时间，不包含在 kernel 时间中；H2D 传输、参考计算和验证也不包含。结果见 `results/tac_bcsc/`，这是经上述兼容性修正的复现结果，不是未经修改的原程序运行结果。
