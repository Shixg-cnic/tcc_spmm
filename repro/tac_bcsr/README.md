# TAC BCSR 行格式独立复现

复现 `TAC_src/TAC/training_backend/c++_frontend/3rd/spmm_cnic_gat_edit_denseC/src/spmm_kernel.cu` 中的 `tf32_computeX128TransposePipe2_BCSR_forward_first_layer`。`original_kernel.cuh` 保留该函数原文；不链接项目 SpMM kernel、PyTorch、IPC 或采样服务。

## 格式与执行

- 一个 CTA 对应固定 8 个输出行。
- 对窗口内源列排序、去重，每 8 个源列组成一个 8×8 tile。
- bitmap 位号为 `output_row % 8 * 8 + local_source_slot`；映射数组保存源节点 ID。
- 原 kernel 双缓冲 bitmap 与源列索引，预取特征，以 TF32 MMA 计算。
- 与 BCSC 的逐 tile 原子输出不同，BCSR 在寄存器中累加整个行窗口，最后用 `st.global.wt.f32` 直接覆盖输出。

## 必要适配

- 复用 `../tac_bcsc/ptx_compat.cuh` 的 PTX 寄存器兼容性修正，RNA TF32 舍入和 MMA 不变；另补齐原 helper 的 write-through store 指令。
- 空行窗口补一个零 bitmap tile，避免原 kernel 无条件预取越界，同时确保空行输出为零。
- 源列 padding 使用 `UINT32_MAX`：原 kernel 后续预取用 `> numCols`，不能把 `numCols` 当作 padding 哨兵。
- 原 kernel 的输出未检查尾部行号，因此输出 buffer 向上补齐到 8 行；验证时只取真实行。
- 仅接受单位边权；F 为 `[64,512]` 内的 32 倍数，沿用原 dispatcher 的 `block=(32,F/32)`。
- kernel 主体与导入源码不作修改。

## 构建和验证

仓库根目录执行：

```bash
mkdir -p build/tmp
TMPDIR="$PWD/build/tmp" cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=120
TMPDIR="$PWD/build/tmp" cmake --build build --target tac_bcsr_repro -j4
ctest --test-dir build -R '^tac_bcsr_correctness$' --output-on-failure
build/tac_bcsr_repro data/ogbn_arxiv_full.csr 128 data/ogbn_arxiv_full.x.bin
build/tac_bcsr_repro data/ogbn_products_full.csr 256 data/ogbn_products_full.x.bin
```

GPU 固定为逻辑设备 0，可用 `CUDA_VISIBLE_DEVICES` 选择物理 GPU。不传 X 文件则使用固定种子随机特征。25 组测试覆盖 F=64/96/128/256/512、空图、空行窗口、单元素、尾部行列、多 tile 及不同流水线奇偶轮次，并拒绝带权输入。小图同时对照 CPU/GPU CSR；全图对照逐行 GPU CSR、TF32 RNA 输入和 double 累加的全部输出。

验证前将输出填充为 NaN，再调用 kernel；另验证不清零重复调用，确保所有真实行均被覆盖。BCSR 的独立执行无需清零，故 `kernel_ms` 为其自然执行成本；`clear_kernel_ms` 额外测量清零开销，便于对齐先前 BCSC 的统计口径。两者均为 CUDA Graph 每组 10 次调用、5 组取中位数，预热 10 次；不含 CPU 格式转换、H2D 或参考计算。结果见 `results/tac_bcsr/`。
